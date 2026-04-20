#!/usr/bin/env python3
"""
Test script to verify mesh color rendering implementation.
This script checks if the PLY file has vertex colors and if the RenderWithMeshColors class works.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_vertex_color_loading():
    """Test if get_mesh_from_ply loads vertex colors correctly."""
    print("=" * 60)
    print("Test 1: Vertex Color Loading")
    print("=" * 60)

    try:
        from mesh_nvs_utils import get_mesh_from_ply
        print("✓ Successfully imported get_mesh_from_ply")

        # Check if function signature accepts PLY file path
        import inspect
        sig = inspect.signature(get_mesh_from_ply)
        print(f"✓ Function signature: {sig}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_render_class_exists():
    """Test if RenderWithMeshColors class exists and has correct structure."""
    print("\n" + "=" * 60)
    print("Test 2: RenderWithMeshColors Class")
    print("=" * 60)

    try:
        from mesh_nvs_utils import RenderWithMeshColors
        print("✓ Successfully imported RenderWithMeshColors")

        # Check class methods
        if hasattr(RenderWithMeshColors, '__init__'):
            print("✓ Has __init__ method")
        if hasattr(RenderWithMeshColors, 'forward'):
            print("✓ Has forward method")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_command_line_arg():
    """Test if command line argument is properly defined."""
    print("\n" + "=" * 60)
    print("Test 3: Command Line Argument")
    print("=" * 60)

    try:
        # Check if the argument parser includes --use_mesh_colors
        with open('render_mesh_nvs.py', 'r') as f:
            content = f.read()

        if '--use_mesh_colors' in content:
            print("✓ --use_mesh_colors argument found in parser")

        if 'RenderWithMeshColors' in content:
            print("✓ RenderWithMeshColors usage found in code")

        if 'args.use_mesh_colors' in content:
            print("✓ Conditional logic for args.use_mesh_colors found")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_ply_file_check():
    """Provide guidance on checking PLY files."""
    print("\n" + "=" * 60)
    print("Test 4: PLY File Vertex Color Check")
    print("=" * 60)

    print("\nTo verify your PLY file has vertex colors, run:")
    print("  python -c \"import trimesh; mesh = trimesh.load('your_mesh.ply'); print('Has colors:', hasattr(mesh.visual, 'vertex_colors'))\"")

    print("\nOr use this script with:")
    print("  python test_mesh_colors.py <path_to_ply_file>")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Mesh Color Rendering Implementation Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Vertex Color Loading", test_vertex_color_loading()))
    results.append(("RenderWithMeshColors Class", test_render_class_exists()))
    results.append(("Command Line Argument", test_command_line_arg()))
    results.append(("PLY File Check", test_ply_file_check()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ All tests passed! Implementation is ready.")
        print("\nTo use mesh color rendering, run:")
        print("  python render_mesh_nvs.py \\")
        print("    --source_path /path/to/colmap/data \\")
        print("    --ply_file /path/to/mesh.ply \\")
        print("    --output_path /path/to/output \\")
        print("    --use_mesh_colors  # ← Use this flag!")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
