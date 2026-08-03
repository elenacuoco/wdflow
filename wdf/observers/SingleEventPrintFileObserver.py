__author__ = "Elena Cuoco"
__copyright__ = "Copyright 2017, Elena Cuoco"
__credits__ = []
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Elena Cuoco"
__email__ = "elena.cuoco@unibo.it"
__status__ = "Development"

import os.path

import pyarrow as pa
import pyarrow.parquet as pq

from wdf.observers.observer import Observer

META_COLUMNS = [
    "gps", "gpsPeak", "duration", "EnWDF", "snrMean", "snrPeak",
    "freqMin", "freqMean", "freqMax", "freqPeak", "wave",
]


class SingleEventPrintTriggers(Observer):
    """Collects triggers found by WDF and writes them to a Parquet file, one
    row per trigger.

    `fullPrint` controls which columns are written in addition to
    `META_COLUMNS`: 0 -- metadata only; 1 -- also the wavelet coefficients
    (`wt0..wtN`); 2 -- also the reconstructed waveform (`rw0..rwN`) instead
    of the coefficients; 3 -- both.

    Rows are buffered and flushed to disk every `flush_every` triggers, so
    memory use stays bounded without needing the whole segment's output held
    in memory at once. `close()` must be called once the segment/stream ends
    to flush any remaining buffered rows and finalize the file -- unlike a
    CSV, a Parquet file's footer is only written on close.
    """

    def __init__(self, par, fullPrint=0, flush_every=500):
        """
        :type par: dict
        :param par: The dictionary of WDF parameters

        :type fullPrint: int
        :param fullPrint: Flag for the output type: 0 - metadata, 1 - wavelet coefficients, 2 - reconstructed waveform, 3 - both

        :type flush_every: int
        :param flush_every: number of buffered trigger rows per Parquet row-group flush
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

        self.fullPrint = fullPrint
        self.flush_every = flush_every

        self.columns = list(META_COLUMNS)
        if fullPrint in (1, 3):
            self.columns += ["wt" + str(i) for i in range(par.Ncoeff)]
        if fullPrint in (2, 3):
            self.columns += ["rw" + str(i) for i in range(par.Ncoeff)]

        self._writer = None
        self._rows = []

    def update(self, CEV):
        """Buffers one trigger; flushes to disk once `flush_every` are queued.

        :type CEV: pytsa object
        :param CEV: object that contains metadata, wavelet coefficients and reconstructed waveform of the trigger.
        """
        ev = CEV.__dict__
        self._rows.append({k: ev[k] for k in self.columns})
        if len(self._rows) >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.filesave, table.schema)
        self._writer.write_table(table)
        self._rows = []

    def close(self):
        """Flushes any remaining buffered triggers and finalizes the file."""
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
