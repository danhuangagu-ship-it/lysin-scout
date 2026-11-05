"""Generate full-length embeddings using the original screening conventions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .predict import MODELS

SIGNATURES = {
    'ESMC': '39cad7a44d640265c820289a3c0b2c55be3adbdad2e852c0e2796c4042cde439',
    'ProtBERT': 'ba0ec04d559698cc50d59a1f2f94f768414542097f67065f200d9760a4684a25',
    'ProSE': 'fa01aadf79fac1df221f60ceafaed501c56edcea3928aacdc4f614a418cb087d'}
DIMS = {'ESMC': 960, 'ProtBERT': 1024, 'ProSE': 6165}


def read_fasta(path):
    records = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            if not line[1:].split():
                raise ValueError('Empty FASTA identifier')
            records.append([line[1:].split()[0], ''])
        elif line.strip():
            if not records:
                raise ValueError('Sequence found before a FASTA header')
            records[-1][1] += ''.join(line.split()).upper()
    if not records or len({r[0] for r in records}) != len(records):
        raise ValueError('FASTA must contain nonempty, unique protein IDs')
    for identifier, sequence in records:
        if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWYBXZJUO*'):
            raise ValueError('Empty sequence or unsupported residue: ' + identifier)
    return records


def encoder_signature(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def load_encoder(name, location, prose_source, device):
    import torch
    if name == 'ESMC':
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein
        from esm.pretrained import get_esmc_model_tokenizers
        model = ESMC(d_model=960, n_heads=15, n_layers=30,
                     tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False).float()
        model.load_state_dict(torch.load(location, map_location='cpu', weights_only=True), strict=True)
        model = model.to(device).eval()
        def encode(sequence):
            tokens = model.encode(ESMProtein(sequence=sequence)).sequence.unsqueeze(0)
            return model(sequence_tokens=tokens).embeddings.squeeze(0).mean(0)
    elif name == 'ProtBERT':
        from transformers import BertModel, BertTokenizer
        tok = BertTokenizer.from_pretrained(location, do_lower_case=False, local_files_only=True)
        model = BertModel.from_pretrained(location, local_files_only=True).float().to(device).eval()
        def encode(sequence):
            inputs = {k: v.to(device) for k, v in tok(' '.join(sequence), return_tensors='pt', truncation=False).items()}
            hidden = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            return ((hidden * mask).sum(1) / mask.sum(1))[0]
    else:
        if not prose_source:
            raise ValueError('--prose-source must point to the upstream ProSE checkout')
        sys.path.insert(0, str(Path(prose_source).resolve()))
        from prose.models.multitask import ProSEMT
        from prose.alphabets import Uniprot21
        model = ProSEMT.load_pretrained(location).float().to(device).eval()
        alphabet = Uniprot21()
        def encode(sequence):
            tokens = torch.from_numpy(alphabet.encode(sequence.encode())).long()[None, :].to(device)
            return model.transform(tokens).squeeze(0).mean(0)
    actual = encoder_signature(model)
    if actual != SIGNATURES[name]:
        raise ValueError(f'{name} encoder fingerprint differs from the trained classifier: {actual}')
    return encode


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--encoder-path', required=True, help='ESMC/ProSE weight file, or local ProtBERT directory')
    parser.add_argument('--prose-source', help='Upstream ProSE checkout; required for ProSE')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        parser.error('Output directory exists; choose a new path')
    records = read_fasta(args.fasta)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encode = load_encoder(args.model, args.encoder_path, args.prose_source, args.device)
    vectors, mapping, cache = [], [], {}
    with torch.inference_mode():
        for identifier, sequence in records:
            effective = sequence.rstrip('*') if args.model == 'ProtBERT' else sequence
            if not effective:
                raise ValueError('No residues remain after terminal-stop removal: ' + identifier)
            if effective not in cache:
                vector = encode(effective).detach().cpu().numpy().astype(np.float32)
                if vector.shape != (DIMS[args.model],) or not np.isfinite(vector).all():
                    raise ValueError('Invalid encoder output: ' + identifier)
                cache[effective] = vector
            vectors.append(cache[effective])
            mapping.append({'id': identifier, 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
                            'encoded_sequence_sha256': hashlib.sha256(effective.encode()).hexdigest()})
    out.mkdir(parents=True)
    np.save(out / 'features.npy', np.stack(vectors), allow_pickle=False)
    with (out / 'ids.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    (out / 'manifest.json').write_text(json.dumps({'model': args.model, 'proteins': len(records),
        'dimension': DIMS[args.model], 'encoder_parameter_sha256': SIGNATURES[args.model],
        'full_sequences': True, 'truncated': False, 'device': args.device}, indent=2))


if __name__ == '__main__':
    main()
