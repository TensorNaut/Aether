import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.pipeline import PSREnhancePipeline

def main():
    parser = argparse.ArgumentParser(description="AETHER - Enhance OHRC PSR Images")
    parser.add_argument('--input', type=str, required=True, help="Path to raw OHRC .xml label")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to trained Zero-DCE .pth model")
    parser.add_argument('--output', type=str, required=True, help="Output image path (.png)")
    parser.add_argument('--y-start', type=int, default=10000, help="Y start pixel coordinate")
    parser.add_argument('--y-end', type=int, default=12000, help="Y end pixel coordinate")
    parser.add_argument('--x-start', type=int, default=0, help="X start pixel coordinate")
    parser.add_argument('--x-end', type=int, default=4000, help="X end pixel coordinate")
    parser.add_argument('--tile-size', type=int, default=512, help="Inference tile size")
    parser.add_argument('--overlap', type=int, default=64, help="Tile overlap for blending")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" AETHER Inference Pipeline")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Run pipeline
    pipeline = PSREnhancePipeline(args.checkpoint)
    
    original, enhanced = pipeline.enhance_crop(
        image_path=args.input,
        y_start=args.y_start,
        y_end=args.y_end,
        x_start=args.x_start,
        x_end=args.x_end,
        tile_size=args.tile_size,
        overlap=args.overlap
    )
    
    # Save side-by-side comparison
    pipeline.save_comparison(original, enhanced, args.output)
    print(f"\nDone! View result at: {args.output}")

if __name__ == '__main__':
    main()
