# -*- coding: utf-8 -*-

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS


class SummarizationModel(object):
    """
    A class to represent a sequence-to-sequence model for text summarization. 
    Supports both baseline mode, pointer-generator mode, and coverage
    """

    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """
        Add placeholders to the graph. 
        These are entry points for any input data.
        """
        hps = self._hps

        # encoder part
        self._enc_batch = tf.placeholder(
            tf.int32, [hps.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(
            tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, None], name='enc_padding_mask')

        # query part
        self._query_batch = tf.placeholder(
            tf.int32, [hps.batch_size, None], name='query_batch')
        self._query_lens = tf.placeholder(
            tf.int32, [hps.batch_size], name='query_lens')
        self._query_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, None], name='query_padding_mask')
        
        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(
                tf.int32, [hps.batch_size, None],
                name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(
                tf.int32, [], name='max_art_oovs')
            self._query_batch_extend_vocab = tf.placeholder(
                tf.int32, [hps.batch_size, None],
                name='query_batch_extend_vocab')

        # decoder part
        self._dec_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, hps.max_dec_steps],
            name='dec_padding_mask')

        if hps.mode == "decode" and hps.coverage:
            self.prev_t_coverage = tf.placeholder(
                tf.float32, [hps.batch_size, None], name='prev_t_coverage')
            self.prev_b_coverage = tf.placeholder(
                tf.float32, [hps.batch_size, None], name='prev_b_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
            batch: Batch object
            just_enc: Boolean. If True(decode mode), only feed the parts needed for the encoder.
        """
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask

        feed_dict[self._query_batch] = batch.query_batch
        feed_dict[self._query_lens] = batch.query_lens
        feed_dict[self._query_padding_mask] = batch.query_padding_mask

        feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
        feed_dict[self._query_batch_extend_vocab] = batch.query_batch_extend_vocab
        feed_dict[self._max_art_oovs] = batch.max_art_oovs

        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len, name=None, reuse=False):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
            encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
            seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
            encoder_outputs:
                A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. 
                It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
            fw_state, bw_state:
                Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope(name or "encoder", reuse=reuse):
            cell_fw = tf.contrib.rnn.LSTMCell(
                self._hps.hidden_dim,
                initializer=self.rand_unif_init,
                state_is_tuple=True)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw,
                input_keep_prob=1.0,
                output_keep_prob=1.0,
                state_keep_prob=1.0)
            if self._hps.encoder_type == 'bi':
                cell_bw = tf.contrib.rnn.LSTMCell(
                    self._hps.hidden_dim,
                    initializer=self.rand_unif_init,
                    state_is_tuple=True)
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw,
                    input_keep_prob=1.0,
                    output_keep_prob=1.0,
                    state_keep_prob=1.0)
                (encoder_outputs, (fw_st, bw_st)) = \
                    tf.nn.bidirectional_dynamic_rnn(
                        cell_fw,
                        cell_bw,
                        encoder_inputs,
                        dtype=tf.float32,
                        sequence_length=seq_len,
                        swap_memory=True)
                # concatenate the forwards and backwards states
                encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
                state = self._reduce_states(fw_st, bw_st, 'reduce_states', reuse=reuse)
            elif self._hps.encoder_type == 'uni':
                (encoder_outputs, state) = \
                    tf.nn.dynamic_rnn(
                        cell_fw,
                        encoder_inputs,
                        dtype=tf.float32,
                        sequence_length=seq_len,
                        swap_memory=True)

        return encoder_outputs, state


    def _reduce_states(self, fw_st, bw_st, name=None, reuse=False):
        """Add to the graph a linear layer to reduce the encoder's 
        final FW and BW state into a single initial state for the decoder. 
        This is needed because the encoder is bidirectional but the decoder is not.

        Args:
            fw_st: LSTMStateTuple with hidden_dim units.
            bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
            state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = self._hps.hidden_dim
        with tf.variable_scope(name or "encoder", reuse=reuse):

            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable(
                'w_reduce_c', [hidden_dim * 2, hidden_dim],
                dtype=tf.float32,
                initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable(
                'w_reduce_h', [hidden_dim * 2, hidden_dim],
                dtype=tf.float32,
                initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable(
                'bias_reduce_c', [hidden_dim],
                dtype=tf.float32,
                initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable(
                'bias_reduce_h', [hidden_dim],
                dtype=tf.float32,
                initializer=self.trunc_norm_init)

            # Apply linear layer
            # Concatenation of fw and bw cell
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
            # Concatenation of fw and bw state
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) +
                               bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) +
                               bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(
                new_c, new_h)  # Return new cell and state

    def _add_decoder(self, inputs):
        """
        Add attention decoder to the graph. 
        In train or eval mode, you call this once to get output on ALL steps.
        In decode (beam search) mode, you call this once for EACH decoder step.

        Args:
            inputs: inputs to the decoder (word embeddings). 
            A list of tensors shape (batch_size, emb_dim)

        Returns:
            outputs: List of tensors; the outputs of the decoder
            out_state: The final state of the decoder
            attn_dists: A list of tensors; the attention distributions
            coverage: A tensor, the current coverage vector
        """
        hps = self._hps
        cell = tf.contrib.rnn.LSTMCell(
            hps.hidden_dim,
            state_is_tuple=True,
            initializer=self.rand_unif_init)
        cell = tf.contrib.rnn.DropoutWrapper(cell, 
            input_keep_prob=1.0,
            output_keep_prob=1.0,
            state_keep_prob=1.0)

        # In decode mode, we run attention_decoder one step at a time
        # and so need to pass in the previous step's coverage vector each time
        if hps.mode == "decode" and hps.coverage:
            prev_t_coverage = self.prev_t_coverage
            prev_b_coverage = self.prev_b_coverage 
        else:
            prev_t_coverage = None
            prev_b_coverage = None

        #todo: 添加 query， 
        outputs, out_state, context_attn_dists, query_attn_dists, p_ts, p_bs, t_coverage, b_coverage = attention_decoder(
            inputs,
            self._dec_in_state,
            self._enc_states,
            self._enc_padding_mask,
            #self._query_rep,
            self._query_states,
            self._query_padding_mask,
            cell,
            initial_state_attention=(hps.mode == "decode"),
            pointer_gen=hps.pointer_gen,
            use_coverage=hps.coverage,
            prev_t_coverage=prev_t_coverage,
            prev_b_coverage=prev_b_coverage)

        return outputs, out_state, context_attn_dists, query_attn_dists, p_ts, p_bs, t_coverage, b_coverage

    def _calc_final_dist(self, context_attn_dists, query_attn_dists):
        """Calculate the final distribution, for the pointer-generator model

        Args:
            vocab_dists: The vocabulary distributions. 
            List length max_dec_steps of (batch_size, vsize) arrays. 
            The words are in the order they appear in the vocabulary file.
            attn_dists: The attention distributions. 
            List length max_dec_steps of (batch_size, attn_len) arrays

        Returns:
            final_dists: The final distributions. 
            List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
        with tf.variable_scope('final_distribution'):
            # context_attn_dists = [tf.multiply(tf.slice(p_t, [0, 0], [-1, 1]), dist)
            #             for (p_t, dist) in zip(self.p_ts, context_attn_dists)]
            # query_attn_dists = [tf.multiply(tf.slice(p_t, [0, 1], [-1, 1]), dist)
            #             for (p_t, dist) in zip(self.p_ts, query_attn_dists)]

            # context_attn_dists = [tf.multiply(p_t, dist)
            #             for (p_t, dist) in zip(self.p_ts, context_attn_dists)]
            # query_attn_dists = [tf.multiply(p_b, dist)
            #             for (p_b, dist) in zip(self.p_bs, query_attn_dists)]

            # context_attn_dists = [tf.slice(p_t, [0, 0], [-1, 1]) * dist
            #             for (p_t, dist) in zip(self.p_ts, context_attn_dists)]
            # query_attn_dists = [tf.slice(p_t, [0, 1], [-1, 1]) * dist
            #             for (p_t, dist) in zip(self.p_ts, query_attn_dists)]

            # Concatenate some zeros to each vocabulary dist,
            # to hold the probabilities for in-article OOV words
            # the maximum (over the batch) size of the extended vocabulary
            extended_vsize = self._vocab.size() + self._max_art_oovs

            # Project the values in the attention distributions onto the appropriate entries in the final distributions
            # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
            # then we add 0.1 onto the 500th entry of the final distribution
            # This is done for each decoder timestep.
            # This is fiddly; we use tf.scatter_nd to do the projection

            # shape (batch_size)
            batch_nums = tf.range(0, limit=self._hps.batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            # number of states we attend over
            context_attn_len = tf.shape(self._enc_batch_extend_vocab)[1]
            # shape (batch_size, attn_len)
            context_batch_nums = tf.tile(batch_nums, [1, context_attn_len])
            # shape (batch_size, enc_t, 2)
            context_indices = tf.stack((context_batch_nums, self._enc_batch_extend_vocab), axis=2)
            shape = [self._hps.batch_size, extended_vsize]
            # list length max_dec_steps (batch_size, extended_vsize)
            context_attn_dists_projected = [
                tf.scatter_nd(context_indices, copy_dist, shape)
                for copy_dist in context_attn_dists
            ]

            query_attn_len = tf.shape(self._query_batch_extend_vocab)[1]
            # shape (batch_size, attn_len)
            query_batch_nums = tf.tile(batch_nums, [1, query_attn_len])
            # shape (batch_size, enc_t, 2)
            query_indices = tf.stack((query_batch_nums, self._query_batch_extend_vocab), axis=2)

            query_attn_dists_projected = [
                tf.scatter_nd(query_indices, copy_dist, shape)
                for copy_dist in query_attn_dists
            ]

            final_dists = [
                context_dist + query_dist
                for (context_dist, query_dist) \
                in zip(context_attn_dists_projected, query_attn_dists_projected)
            ]

            return final_dists

    def _add_emb_vis(self, embedding_var):
        """
        Do setup so that we can view word embedding visualization in Tensorboard,
        as described here:
        https://www.tensorflow.org/get_started/embedding_viz
        Make the vocab metadata file, then make the projector config file pointing to it.
        """
        train_dir = os.path.join(FLAGS.log_root, "train")
        vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
        self._vocab.write_metadata(vocab_metadata_path)  # write metadata file
        summary_writer = tf.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)

    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self._hps
        vsize = self._vocab.size()

        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(
                -hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=42)
            self.trunc_norm_init = tf.truncated_normal_initializer(
                stddev=hps.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable(
                    'embedding', [vsize, hps.emb_dim],
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init)
                if hps.mode == "train":
                    self._add_emb_vis(embedding)
                # tensor with shape (batch_size, max_enc_steps, emb_size)
                emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
                emb_query_inputs = tf.nn.embedding_lookup(embedding, self._query_batch)

                # list length max_dec_steps containing shape (batch_size, emb_size)
                emb_dec_inputs = [
                    tf.nn.embedding_lookup(embedding, x)
                    for x in tf.unstack(self._dec_batch, axis=1)
                ]

            # Add the encoder.
             # Add the encoder.
            enc_outputs, context_state = self._add_encoder(emb_enc_inputs, self._enc_lens, 'encoder')
            
            #todo: Add the query encoder.
            query_outputs, query_state = self._add_encoder(emb_query_inputs, self._query_lens, 'encoder', True)

            self._enc_states = enc_outputs
            self._query_states = query_outputs
            self._query_rep = query_state
            self._dec_in_state = self._reduce_states(context_state, query_state, 'reduce_final_st')

            # Add the decoder.
            with tf.variable_scope('decoder'):
                decoder_outputs, self._dec_out_state, self.context_attn_dists, \
                self.query_attn_dists, self.p_ts, self.p_bs, \
                self.t_coverage, self.b_coverage = self._add_decoder(emb_dec_inputs)

                # self.query_attn_dists = [tf.multiply(p_b, dist)
                #         for (p_b, dist) in zip(self.p_bs, self.query_attn_dists)]

            # For pointer model, calc final distribution from copy distribution
            final_dists = self._calc_final_dist(self.context_attn_dists, self.query_attn_dists)

        self.final_dists = final_dists

        if hps.mode == "decode":
            if hps.batch_size > 0:
                # We run decode beam search mode one decoder step at a time
                # final_dists is a singleton list containing shape (batch_size, extended_vsize)
                assert len(final_dists) == 1
                final_dists = final_dists[0]
                # take the k largest probs. note batch_size=beam_size in decode mode
                topk_probs, self._topk_ids = tf.nn.top_k(
                    final_dists, hps.batch_size * 2)
                self._topk_log_probs = tf.log(topk_probs)
            else:
                assert len(final_dists) == 1
                # take the k largest probs
                topk_probs, self._topk_ids = tf.nn.top_k(final_dists, k=1)
                self._topk_log_probs = tf.log(topk_probs)

    def _add_loss(self):
        with tf.variable_scope('loss'):
            if FLAGS.pointer_gen:
                # Calculate the loss per step
                # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
                # will be list length max_dec_steps containing shape (batch_size)
                loss_per_step = []
                # shape (batch_size)
                batch_nums = tf.range(0, limit=self._hps.batch_size)
                for dec_step, dist in enumerate(self.final_dists):
                    # The indices of the target words. shape (batch_size)
                    targets = self._target_batch[:, dec_step]
                    # shape (batch_size, 2)
                    indices = tf.stack((batch_nums, targets), axis=1) 
                    # shape (batch_size). prob of correct words on this step
                    gold_probs = tf.gather_nd(dist, indices)
                    losses = -tf.log(tf.clip_by_value(gold_probs, 1e-10, 1.0))
                    loss_per_step.append(losses)

                # Apply dec_padding_mask and get loss
                self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

            tf.summary.scalar('loss', self._loss)

            # Calculate coverage loss from the attention distributions
            if self._hps.coverage:
                with tf.variable_scope('coverage_loss'):
                    t_coverage_loss = _coverage_loss(
                        self.context_attn_dists, self._dec_padding_mask)
                    b_coverage_loss = _coverage_loss(
                        self.query_attn_dists, self._dec_padding_mask)

                    self._coverage_loss = t_coverage_loss + b_coverage_loss
                    tf.summary.scalar('coverage_loss', self._coverage_loss)
                self._total_loss = self._loss + self._hps.cov_loss_wt * self._coverage_loss
                tf.summary.scalar('total_loss', self._total_loss)

    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(
            loss_to_minimize,
            tvars,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        grads, global_norm = tf.clip_by_global_norm(gradients,
                                                    self._hps.max_grad_norm)

        # Add a summary
        tf.summary.scalar('global_norm', global_norm)

        optimizer = tf.train.AdagradOptimizer(
            self._hps.learning_rate,
            initial_accumulator_value=self._hps.adagrad_init_acc)

        # optimizer = tf.train.AdamOptimizer(self._hps.learning_rate)

        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name='train_step')

    def build_graph(self):
        """
        Add the placeholders, model, global step, train_op and summaries to the graph
        """
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        self._add_seq2seq()
        if self._hps.mode in ['train', 'eval']:
            self._add_loss()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

        Args:
            sess: Tensorflow session.
            batch: Batch object that is the same example repeated across the batch (for beam search)

        Returns:
            enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
            dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
        """
        # feed the batch into the placeholders
        feed_dict = self._make_feed_dict(batch, just_enc=True)
        (enc_states, query_states, dec_in_state, global_step) = sess.run(
            [self._enc_states, self._query_states, self._dec_in_state, self.global_step],
            feed_dict)

        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        if FLAGS.mode == 'decode' and FLAGS.beam_size > 0:
            dec_in_state = tf.contrib.rnn.LSTMStateTuple(
                dec_in_state.c[0], dec_in_state.h[0])
        else:
            dec_in_state = tf.contrib.rnn.LSTMStateTuple(
                dec_in_state.c, dec_in_state.h)

        return enc_states, query_states, dec_in_state

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, query_states,
                       dec_init_states, prev_t_coverage, prev_b_coverage):
        """For beam search decoding. Run the decoder for one step.

        Args:
            sess: Tensorflow session.
            batch: Batch object containing single example repeated across the batch
            latest_tokens: Tokens to be fed as input into the decoder for this timestep
            enc_states: The encoder states.
            dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
            prev_coverage: List of np arrays. The coverage vectors from the previous timestep. 
            List of None if not using coverage.

        Returns:
            ids: top 2k ids. shape [beam_size, 2*beam_size]
            probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
            new_states: new states of the decoder. a list length beam_size containing
                LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
            attn_dists: List length beam_size containing lists length attn_length.
            new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
        """

        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [
            np.expand_dims(state.h, axis=0) for state in dec_init_states
        ]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(
            hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._query_states: query_states,
            self._query_padding_mask: batch.query_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.context_attn_dists, #todo: 需要修改，可视化对应部分，需要改为context，query
            #"query_attn_dists": self.query_attn_dists,
            "p_ts": self.p_ts
        }

        feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
        feed[self._query_batch_extend_vocab] = batch.query_batch_extend_vocab
        feed[self._max_art_oovs] = batch.max_art_oovs

        if self._hps.coverage:
            feed[self.prev_t_coverage] = np.stack(prev_t_coverage, axis=0)
            feed[self.prev_b_coverage] = np.stack(prev_b_coverage, axis=0)
            to_return['t_coverage'] = self.t_coverage
            to_return['b_coverage'] = self.b_coverage

        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [
            tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :],
                                          results['states'].h[i, :])
            for i in xrange(beam_size)
        ]

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if FLAGS.coverage:
            new_t_coverage = results['t_coverage'].tolist()
            new_b_coverage = results['b_coverage'].tolist()
            assert len(new_t_coverage) == beam_size
        else:
            new_t_coverage = [None for _ in xrange(beam_size)]
            new_b_coverage = [None for _ in xrange(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, new_t_coverage, new_b_coverage


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
        values: a list length max_dec_steps containing arrays shape (batch_size).
        padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
        a scalar
    """
    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [
        v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)
    ]
    # shape (batch_size); normalized value for each batch member
    values_per_ex = sum(values_per_step) / dec_lens
    return tf.reduce_mean(values_per_ex)  # overall average


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
        attn_dists: The attention distributions for each decoder timestep. 
        A list length max_dec_steps containing shape (batch_size, attn_length)
        padding_mask: shape (batch_size, max_dec_steps).

    Returns:
        coverage_loss: scalar
    """
    # shape (batch_size, attn_length). Initial coverage is zero.
    coverage = tf.zeros_like(attn_dists[0])
    # Coverage loss per decoder timestep. 
    # Will be list length max_dec_steps containing shape (batch_size).
    covlosses = []  
    for a in attn_dists:
        # calculate the coverage loss for this step
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss
