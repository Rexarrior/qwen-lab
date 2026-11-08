"""Apply one of the release precision maps to an original F32 GGUF."""
import argparse, hashlib, subprocess
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('policy', choices=['q4-q6-q8', 'protected-iq3'])
    p.add_argument('source', type=Path)
    p.add_argument('imatrix', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--binary', default='llama-quantize')
    args = p.parse_args()
    if args.output.exists():
        p.error('output already exists')
    if hashlib.sha256(args.imatrix.read_bytes()).hexdigest() != '8eeffbe08370c5cc1a27adb534ff7940094691b18d406f7bb624f3fc0e08f39e':
        p.error('importance matrix does not match the release input')
    mode, embedding = {'q4-q6-q8': ('Q4_K_M','Q8_0'), 'protected-iq3': ('IQ3_S','Q5_K')}[args.policy]
    table = Path(__file__).with_name(args.policy + '.types')
    subprocess.run([args.binary, '--imatrix', args.imatrix, '--pure', '--tensor-type-file', table,
                    '--token-embedding-type', embedding, args.source, args.output, mode, '8'], check=True)

if __name__ == '__main__':
    main()
