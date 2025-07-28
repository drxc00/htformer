# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['real_time_demo.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models/final/hierarchical_transformer_f201_d64_h2_s1_t1_do0.1_20250701_2251.pth', 'models/final'),
        ('models/mediapipe/pose_landmarker_full.task', 'models/mediapipe'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='real_time_demo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
