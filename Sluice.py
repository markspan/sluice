"""
sluice.py  –  XDF → BrainVision Analyzer converter
Requires: pip install mne pyxdf PySide6
"""

import sys
import traceback
from pathlib import Path

import pyxdf
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QTextEdit, QGroupBox,
    QAbstractItemView, QMessageBox, QFrame
)


# ─── Worker thread ────────────────────────────────────────────────────────────

class ConverterWorker(QThread):
    log     = Signal(str)
    success = Signal(str)
    error   = Signal(str)

    def __init__(self, xdf_path, eeg_stream_id, marker_stream_ids):
        super().__init__()
        self.xdf_path          = xdf_path
        self.eeg_stream_id     = eeg_stream_id
        self.marker_stream_ids = marker_stream_ids

    def run(self):
        try:
            import mne

            self.log.emit("Loading XDF file …")
            streams, _ = pyxdf.load_xdf(self.xdf_path)

            # Find chosen stream
            eeg_stream = next(s for s in streams
                              if int(s["info"]["stream_id"]) == self.eeg_stream_id)

            srate = float(eeg_stream["info"]["nominal_srate"][0])
            n_ch  = int(eeg_stream["info"]["channel_count"][0])
            name  = eeg_stream["info"]["name"][0]
            stype = eeg_stream["info"]["type"][0].lower()
            self.log.emit(f"Stream: {name}  |  {n_ch} ch  |  {srate} Hz")

            # Build channel names
            try:
                ch_labels = [
                    ch["label"][0]
                    for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
                ]
            except Exception:
                ch_labels = [f"CH{i+1}" for i in range(n_ch)]

            # Use "eeg" ch_type only for streams that self-identify as EEG and
            # scale µV → V. Everything else (ECG, EMG, power, etc.) gets "misc"
            # with no scaling, avoiding MNE's unit enforcement.
            if stype == "eeg":
                ch_types = "eeg"
                scale    = 1e-6    # LSL EEG streams are typically in µV
                self.log.emit("Channel type: EEG  (scaling µV → V)")
            else:
                ch_types = "misc"
                scale    = 1.0
                self.log.emit(f"Channel type: misc  (raw LSL units, no scaling)")

            data = eeg_stream["time_series"].T * scale   # (n_ch, n_samples)

            info = mne.create_info(
                ch_names=ch_labels,
                sfreq=srate,
                ch_types=ch_types,
            )

            # verbose=False sidesteps the _get_argvalues() KeyError introduced
            # in MNE ≥ 1.7 when running inside onefile/frozen executables.
            raw = mne.io.RawArray(data, info, verbose=False)
            self.log.emit(f"Created RawArray: {data.shape[1]} samples")

            # Add annotations from marker streams
            onsets, durations, descriptions = [], [], []
            eeg_t0 = eeg_stream["time_stamps"][0]

            for stream in streams:
                sid = int(stream["info"]["stream_id"])
                if sid not in self.marker_stream_ids:
                    continue
                sname = stream["info"]["name"][0]
                self.log.emit(f"Adding markers from stream: {sname}")
                for ts, sample in zip(stream["time_stamps"],
                                      stream["time_series"]):
                    onset = ts - eeg_t0
                    if onset < 0:
                        continue
                    onsets.append(onset)
                    durations.append(0.0)
                    descriptions.append(str(sample[0]))

            if onsets:
                raw.set_annotations(
                    mne.Annotations(onsets, durations, descriptions)
                )
                self.log.emit(f"Added {len(onsets)} marker(s)")
            else:
                self.log.emit("No markers found / no marker streams selected.")

            # Output: same directory, same basename, .vhdr
            out_path = str(Path(self.xdf_path).with_suffix(".vhdr"))
            self.log.emit(f"Exporting to {out_path} …")
            mne.export.export_raw(out_path, raw, fmt="brainvision",
                                  overwrite=True, verbose=False)

            self.success.emit(out_path)

        except Exception:
            self.error.emit(traceback.format_exc())


# ─── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.xdf_path = None
        self.streams  = []
        self.worker   = None
        self._setup_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle("Sluice  –  XDF → BrainVision")
        self.setMinimumSize(820, 620)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #1a1d23;
                color: #d4d8e0;
                font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #2e3340;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 600;
                color: #7a8299;
                font-size: 11px;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }

            QPushButton {
                background: #252930;
                border: 1px solid #353a47;
                border-radius: 5px;
                padding: 7px 18px;
                color: #d4d8e0;
            }
            QPushButton:hover   { background: #2e3340; border-color: #4a8fe8; }
            QPushButton:pressed { background: #1f2229; }
            QPushButton:disabled { color: #444; border-color: #2a2d35; }

            QPushButton#primary {
                background: #2563c7;
                border-color: #3b7be0;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#primary:hover    { background: #3071d9; }
            QPushButton#primary:disabled { background: #1c3a6e; color: #5a7aaa; }

            QTableWidget {
                background: #1e2128;
                border: 1px solid #2e3340;
                border-radius: 5px;
                gridline-color: #252930;
            }
            QTableWidget::item { padding: 4px 8px; }
            QTableWidget::item:selected {
                background: #1e3a6e;
                color: #ffffff;
            }
            QHeaderView::section {
                background: #252930;
                border: none;
                border-bottom: 1px solid #2e3340;
                padding: 5px 8px;
                color: #7a8299;
                font-size: 11px;
                letter-spacing: 0.5px;
            }

            QTextEdit {
                background: #12141a;
                border: 1px solid #2e3340;
                border-radius: 5px;
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 12px;
                color: #8fba8f;
            }

            QProgressBar {
                background: #1e2128;
                border: 1px solid #2e3340;
                border-radius: 4px;
                height: 6px;
                text-align: center;
            }
            QProgressBar::chunk { background: #2563c7; border-radius: 3px; }

            QLabel#fileLabel {
                color: #4a8fe8;
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 12px;
            }
            QLabel#hint { color: #555c70; font-size: 12px; }
            QFrame#divider { background: #2e3340; }
        """)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # ── Title ──
        title = QLabel("Sluice  –  XDF → BrainVision")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Light))
        title.setStyleSheet("color:#4a8fe8; letter-spacing:1px;")
        layout.addWidget(title)

        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

        # ── File selection ──
        file_group  = QGroupBox("Source File")
        fg_layout   = QHBoxLayout(file_group)
        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("fileLabel")
        fg_layout.addWidget(self.file_label, 1)
        self.btn_open = QPushButton("Open XDF …")
        self.btn_open.clicked.connect(self.open_file)
        fg_layout.addWidget(self.btn_open)
        layout.addWidget(file_group)

        # ── Stream selection ──
        stream_group = QGroupBox("Stream Selection")
        sg_layout    = QVBoxLayout(stream_group)

        hint = QLabel("Select exactly one signal stream. "
                      "Optionally select marker streams (Ctrl/Shift for multi-select).")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        sg_layout.addWidget(hint)

        sg_layout.addWidget(QLabel("Signal stream:"))
        self.eeg_table = self._make_table(single=True)
        sg_layout.addWidget(self.eeg_table, 2)

        sg_layout.addWidget(QLabel("Marker / Trigger streams (optional):"))
        self.marker_table = self._make_table(single=False)
        sg_layout.addWidget(self.marker_table, 1)

        layout.addWidget(stream_group, 1)

        # ── Progress + log ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(130)
        layout.addWidget(self.log)

        # ── Convert button ──
        self.btn_convert = QPushButton("Convert to BrainVision")
        self.btn_convert.setObjectName("primary")
        self.btn_convert.setEnabled(False)
        self.btn_convert.clicked.connect(self.run_conversion)
        layout.addWidget(self.btn_convert)

    def _make_table(self, single: bool) -> QTableWidget:
        t = QTableWidget(0, 4)
        t.setHorizontalHeaderLabels(["ID", "Name", "Type", "Ch / Rate"])
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        t.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        if single:
            t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            t.itemSelectionChanged.connect(self._update_convert_btn)
        else:
            t.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        return t

    # ── File loading ──────────────────────────────────────────────────────────

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open XDF File", "", "XDF files (*.xdf *.xdfz)")
        if not path:
            return
        self.xdf_path = path
        self.file_label.setText(path)
        self._log(f"Loading stream info from {Path(path).name} …")
        self.btn_open.setEnabled(False)
        try:
            self.streams, _ = pyxdf.load_xdf(path)
            self._populate_tables()
            self._log(f"Found {len(self.streams)} stream(s).")
        except Exception as e:
            self._log(f"ERROR: {e}", error=True)
        finally:
            self.btn_open.setEnabled(True)

    def _populate_tables(self):
        for table in (self.eeg_table, self.marker_table):
            table.setRowCount(0)

        for s in self.streams:
            info   = s["info"]
            sid    = info["stream_id"]
            name   = info["name"][0]
            stype  = info["type"][0]
            n_ch   = info["channel_count"][0]
            srate  = info["nominal_srate"][0]
            detail = f"{n_ch} ch  •  {float(srate):.1f} Hz"
            row_data = [str(sid), name, stype, detail]

            is_signal = (
                stype.lower() not in ("marker", "markers", "trigger",
                                      "triggers", "events", "stimulations")
                and float(srate) > 0
            )
            target = self.eeg_table if is_signal else self.marker_table
            r = target.rowCount()
            target.insertRow(r)
            for col, val in enumerate(row_data):
                item = QTableWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, int(sid))
                target.setItem(r, col, item)

        if self.eeg_table.rowCount() == 1:
            self.eeg_table.selectRow(0)

    # ── Conversion ────────────────────────────────────────────────────────────

    def _update_convert_btn(self):
        self.btn_convert.setEnabled(
            self.xdf_path is not None and
            len(self.eeg_table.selectedItems()) > 0
        )

    def run_conversion(self):
        if not self.eeg_table.selectedItems():
            return
        eeg_sid = self.eeg_table.item(
            self.eeg_table.currentRow(), 0
        ).data(Qt.ItemDataRole.UserRole)

        marker_sids = set()
        for item in self.marker_table.selectedItems():
            marker_sids.add(
                self.marker_table.item(item.row(), 0)
                    .data(Qt.ItemDataRole.UserRole)
            )

        self._log("─" * 50)
        self._log(f"Signal stream ID : {eeg_sid}")
        self._log(f"Marker IDs       : {marker_sids or '(none)'}")

        self.btn_convert.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress.setVisible(True)

        self.worker = ConverterWorker(self.xdf_path, eeg_sid, marker_sids)
        self.worker.log.connect(self._log)
        self.worker.success.connect(self._on_success)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_success(self, out_path):
        self.progress.setVisible(False)
        self.btn_open.setEnabled(True)
        self.btn_convert.setEnabled(True)
        self._log(f"✓ Done! Written to: {out_path}")
        QMessageBox.information(self, "Conversion complete",
                                f"Saved to:\n{out_path}")

    def _on_error(self, tb):
        self.progress.setVisible(False)
        self.btn_open.setEnabled(True)
        self.btn_convert.setEnabled(True)
        self._log("ERROR:\n" + tb, error=True)
        QMessageBox.critical(self, "Conversion failed",
                             "An error occurred. See the log for details.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, error: bool = False):
        colour = "#e05555" if error else "#8fba8f"
        self.log.append(f'<span style="color:{colour}">{msg}</span>')
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum()
        )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())