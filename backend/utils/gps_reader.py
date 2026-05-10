import random
import re
import struct


def _extract_video_gps(video_path):
    """
    Extract GPS coordinates from an MP4/MOV video's ©xyz metadata box.
    The ©xyz atom uses ISO 6709 short format: e.g. '+09.6817+080.0906/'
    Returns (lat, lon) floats or None if not found.
    """
    try:
        with open(video_path, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)
            # Walk top-level boxes to find moov, then scan for ©xyz
            pos = 0
            while pos < file_size:
                f.seek(pos)
                header = f.read(8)
                if len(header) < 8:
                    break
                size = struct.unpack('>I', header[:4])[0]
                name = header[4:8]
                if size == 0:
                    size = file_size - pos
                elif size == 1:
                    ext = f.read(8)
                    size = struct.unpack('>Q', ext)[0]
                if name == b'moov':
                    # Read entire moov box and grep for ©xyz
                    f.seek(pos)
                    moov_data = f.read(size)
                    idx = moov_data.find(b'\xa9xyz')
                    if idx == -1:
                        break
                    # ©xyz box: [4-byte size][©xyz][2-byte len][2-byte lang][string]
                    payload = moov_data[idx + 8:]
                    # strip optional 2-byte length + 2-byte language code prefix
                    raw = payload[:50].decode('latin1', errors='replace')
                    # ISO 6709 pattern: ±DD.DDDD±DDD.DDDD/ (may also have altitude)
                    m = re.search(r'([+-]\d+\.\d+)([+-]\d+\.\d+)', raw)
                    if m:
                        return float(m.group(1)), float(m.group(2))
                    break
                pos += size
    except Exception:
        pass
    return None


class GPSReader:
    """
    GPS input reader.
    If a video_path is supplied and it contains ©xyz metadata, the session is
    anchored to the real recorded location and the reader simulates driving
    movement around that point.
    Falls back to a default Colombo starting point when no metadata is found.
    """

    # Default fallback: Galle Face Green, Colombo
    _DEFAULT_LAT = 6.9271
    _DEFAULT_LON = 79.8612

    def __init__(self, video_path=None):
        anchor_lat, anchor_lon = self._DEFAULT_LAT, self._DEFAULT_LON
        self._source = "simulation (default)"

        if video_path:
            result = _extract_video_gps(video_path)
            if result:
                anchor_lat, anchor_lon = result
                self._source = f"video metadata ({video_path})"
                print(f"[GPS] Extracted from video: lat={anchor_lat}, lon={anchor_lon}")
            else:
                print(f"[GPS] No ©xyz metadata in '{video_path}', using default location.")

        self.lat = anchor_lat
        self.lon = anchor_lon
        # Small driving heading (≈ South-East for default, generic small step otherwise)
        self.d_lat = -0.0001
        self.d_lon = 0.00005

    def get_location(self):
        """Return (lat, lon) advancing along a simulated driving path."""
        self.lat += self.d_lat + random.uniform(-0.00002, 0.00002)
        self.lon += self.d_lon + random.uniform(-0.00002, 0.00002)

        # Slowly drift direction to simulate curves
        if random.random() < 0.1:
            self.d_lat += random.uniform(-0.00001, 0.00001)
            self.d_lon += random.uniform(-0.00001, 0.00001)

        return self.lat, self.lon