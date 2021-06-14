def post_process_one():
    # blocking call that starts a new job
    raise NotImplementedError()


def post_process_remote():
    # non blocking call to start a new job
    raise NotImplementedError()


def post_process_local():
    # actually perform the post processing
    raise NotImplementedError()