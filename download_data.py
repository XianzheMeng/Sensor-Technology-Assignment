"""Download the attributed public ODMR file and verify its SHA-256."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'odmr_public.dat'
URL = 'https://ndownloader.figshare.com/files/53646563'
SHA256 = '748e54dcd66b64ba18999f230b752dfb770d951536102c86fa40bd8160e71ac7'
SIZE = 18_974_276


def valid(path):
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == SHA256


def main():
    if TARGET.exists():
        if not valid(TARGET):
            raise RuntimeError(f'Existing file failed integrity check: {TARGET}')
        print('Public ODMR file already present; SHA-256 verified.')
        return
    temporary = TARGET.with_suffix('.dat.part')
    request = urllib.request.Request(URL, headers={'User-Agent': 'NV-Course-Assignment/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                output.write(chunk)
        if not valid(temporary):
            raise RuntimeError('Downloaded public data failed size or SHA-256 verification.')
        temporary.replace(TARGET)
    finally:
        temporary.unlink(missing_ok=True)
    print('Downloaded public ODMR data; SHA-256 verified. See DATA_LICENSE.md.')


if __name__ == '__main__':
    main()
