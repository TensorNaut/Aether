import sys, csv, os, glob, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from src.data_io.pds4_reader import load_ohrc_image, image_statistics

base = r'D:\Projects and Coding\Version Control Systems\Aether_Data'
dirs = [
    'ch2_ohr_ncp_20250310T0833447498_d_img_d18',
    'ch2_ohr_ncp_20250310T1032168944_d_img_d18',
    'ch2_ohr_ncp_20241218T0948555350_d_img_d18',
    'ch2_ohr_ncp_20241218T1147291344_d_img_d18',
]

for d in dirs:
    sep = '=' * 60
    print(f'\n{sep}')
    print(f' {d}')
    print(sep)

    dpath = os.path.join(base, d)

    # Coordinates from geometry CSV
    csvs = glob.glob(os.path.join(dpath, '**', '*.csv'), recursive=True)
    if csvs:
        with open(csvs[0], 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            lats = [float(r[1]) for r in rows if r[1].strip()]
            lons = [float(r[0]) for r in rows if r[0].strip()]
            print(f'  Lat: {min(lats):.2f} to {max(lats):.2f}')
            print(f'  Lon: {min(lons):.2f} to {max(lons):.2f}')

    # Load and analyze image
    xmls = glob.glob(os.path.join(dpath, '**', '*_d_img_*.xml'), recursive=True)
    if xmls:
        img = load_ohrc_image(xmls[0])
        stats = image_statistics(img)
        shape = stats['shape']
        mean_px = stats['mean']
        dark = stats['pct_dark']
        bright = stats['pct_bright']
        snr = stats['estimated_snr']
        cls = stats['classification']
        print(f'  Shape: {shape}')
        print(f'  Mean px: {mean_px:.1f}/255')
        print(f'  Dark: {dark:.1f}%  Bright: {bright:.1f}%')
        print(f'  SNR: {snr:.3f}')
        print(f'  >> {cls}')
