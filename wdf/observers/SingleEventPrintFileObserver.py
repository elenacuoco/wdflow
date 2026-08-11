"""Writing the search's triggers to disk, one row per trigger.

The observer that terminates the search chain. It receives each trigger as the
parameter-estimation stage builds it and appends it to a Parquet file through
`wdf.analysis.coefficients.TriggerWriter`, in the schema every downstream
consumer reads.

Parquet is written in blocks and its footer only at the end, so the file is
unreadable until the writer is closed. The observer therefore has an explicit
`close()`, and a segment whose run ends without one leaves a file that cannot be
opened at all rather than a short one.
"""
__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"
__status__ = "Development"

import os.path

from wdf.analysis.coefficients import TriggerWriter
from wdf.observers.observer import Observer


class SingleEventPrintTriggers(Observer):
    """Writes the triggers WDF finds to a Parquet file, one row per trigger.

    `close()` must be called once the segment ends: unlike a CSV, a Parquet
    file's footer is only written on close, and a file without one cannot be
    read at all.
    """

    def __init__(self, par, flush_every=500):
        """
        :type par: wdf.config.Parameters.Parameters
        :param par: run configuration, read for the output path.
        :type flush_every: int
        :param flush_every: triggers buffered per Parquet row group.
        """
        self.filesave = par.dir + "WDFTriggers-%s-GPS%s-AR%s-Win%s-Ov%s-EnWDF%s.parquet" % (
            par.channel.replace(':', '-'),
            int(par.gps),
            par.ARorder,
            par.window,
            par.overlap,
            str(par.threshold),
        )
        if os.path.isfile(self.filesave):
            try:
                os.remove(self.filesave)
            except OSError:
                pass

        self.writer = TriggerWriter(self.filesave, flush_every=flush_every)

    def update(self, CEV):
        """Writes one trigger.

        :type CEV: wdf.structures.eventPE.eventPE
        :param CEV: the trigger's record.
        :return: None
        """
        self.writer.append(CEV.record())

    def close(self):
        """Writes what is buffered and finalises the file."""
        self.writer.close()
