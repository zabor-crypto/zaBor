"""REST-driven side-pollers for streams the WS gateway can't deliver.

Lives next to ``venue_clients/`` because functionally it is just another
producer feeding the writer. The supervisor runs each poller as a
co-equal task with WS clients.
"""
