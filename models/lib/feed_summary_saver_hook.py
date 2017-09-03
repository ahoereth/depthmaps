import tensorflow as tf


class FeedSummarySaverHook(tf.train.SummarySaverHook):
    """Saves summaries every N steps using a given feed_dict."""

    def __init__(self, feed_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._feed_dict = feed_dict

        # Maybe some feed_dict values are Tensors which need to be evaluated.
        self._finalized = True
        for value in self._feed_dict.values():
            if isinstance(value, tf.Tensor):
                self._finalized = False

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._request_summary = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step))
        requests = {'global_step': self._global_step_tensor}

        # Evaluate tensors in the feed_dict once and store them.
        if not self._finalized:
            sess = run_context.session
            for key, value in self._feed_dict.items():
                if isinstance(value, tf.Tensor):
                    self._feed_dict[key] = sess.run(value)
            self._finalzied = True

        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):  # pylint: disable=W0613
        if not self._summary_writer:
            return

        global_step = run_values.results['global_step']
        self._next_step = global_step + 1

        if self._request_summary:
            self._timer.update_last_triggered_step(global_step)
            sess = run_context.session
            summaries = sess.run(self._get_summary_op(), self._feed_dict)
            for summary in summaries:
                self._summary_writer.add_summary(summary, global_step)

    def end(self, session):
        """Run a final summary before closing the session."""
        fetches = self._get_summary_op() + [self._global_step_tensor]
        fetched = session.run(fetches, self._feed_dict)
        global_step = fetched[-1]
        for summary in fetched[:-1]:
            self._summary_writer.add_summary(summary, global_step)
