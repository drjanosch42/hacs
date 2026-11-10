// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
//
// (hecke-extensions slot 22, MP-3b) FROZEN: docs/implementation/slot22/mp/mp3b/mp3b_oracle.out
// (sha256 5b85e39381b9fe295d4d4e4f5485fbbb9ef5933bac628ae231107d8eb7029243),
// transcribed mechanically by mp3b_oracle_to_inc.py -- never edit by hand.  Included by MP3bOracle.cpp only.
static const char* const kMP3bOracleSha256 = "5b85e39381b9fe295d4d4e4f5485fbbb9ef5933bac628ae231107d8eb7029243";
static const MP3bGroupRow kGroupRowsB[] = {
    {"G0(2)", 2, 2, 0, {2, 4, 3, 3}},
    {"G0(2)", 2, 2, 1, {1, 3, 3, 3}},
    {"G0(4)", 2, 2, 0, {2, 5, 3, 3}},
    {"G0(4)", 2, 2, 1, {1, 4, 4, 4}},
    {"G0(3)", 3, 2, 0, {2, 3, 1, 1}},
    {"G0(9)", 3, 2, 0, {2, 6, 0, 0}},
    {"G0(5)", 5, 2, 0, {2, 2, 0, 0}},
    {"G0(25)", 5, 2, 0, {2, 10, 0, 0}},
    {"G1(4)", 2, 1, 0, {1, 2, 0, 0}},
    {"G1(4)", 3, 1, 1, {0, 2, 0, 0}},
    {"G1(8)", 2, 1, 0, {1, 5, 0, 0}},
    {"G1(8)", 3, 1, 1, {0, 8, 0, 0}},
};
static const MP3bPairRow kPairRowsB[] = {
    {"G0(2)", "G0(4)", 2, 2, 2, 0, {2, 4, 3, 3}, {2, 5, 3, 3}, {2, 3, 2, 2}, {0, 1, 3, 2}},
    {"G0(2)", "G0(4)", 2, 2, 2, 1, {1, 3, 3, 3}, {1, 4, 4, 4}, {1, 2, 2, 2}, {0, 1, 3, 3}},
    {"G0(3)", "G0(9)", 3, 3, 2, 0, {2, 3, 1, 1}, {2, 6, 0, 0}, {2, 2, 0, 0}, {0, 1, 5, 1}},
    {"G0(5)", "G0(25)", 5, 5, 2, 0, {2, 2, 0, 0}, {2, 10, 0, 0}, {2, 2, 0, 0}, {0, 0, 8, 0}},
    {"Gamma", "Gamma^2", 2, 2, 1, 0, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 0, 1, 0}, {0, 1, 1, 1}},
    {"G0(2)", "G0(4)", 2, 3, 1, 2, {0, 2, 0, 0}, {0, 3, 0, 0}, {0, 2, 0, 0}, {0, 0, 1, 0}},
    {"G0(2)", "G0(4)", 2, 2, 2, 1, {1, 3, 3, 3}, {1, 4, 4, 4}, {1, 2, 2, 2}, {0, 1, 3, 3}},
    {"G1(4)", "G1(8)", 4, 2, 1, 0, {1, 2, 0, 0}, {1, 5, 0, 0}, {1, 1, 0, 0}, {0, 1, 4, 0}},
    {"G1(4)", "G1(8)", 4, 3, 1, 1, {0, 2, 0, 0}, {0, 8, 0, 0}, {0, 2, 0, 0}, {0, 0, 6, 0}},
};
static const MP3bDescRow kDescRowsB[] = {
    {"M7", "G0(5)", 6, true, 24, 30, {30, 762, 15246, 318858}},
    {"M7", "G0(25)", 30, true, 116, 150, {150, 3810, 76230, 1594290}},
    {"M7", "G1(8)", 48, false, 92, 120, {0, 0, 0, 0}},
    {"R_Z", "Gamma^2", 2, true, 11, 14, {14, 330, 5854, 112698}},
    {"R_Z", "G0(2)", 3, true, 15, 21, {21, 495, 8781, 169047}},
    {"R_Z", "G0(4)", 6, true, 29, 42, {42, 990, 17562, 338094}},
};
static const MP3bForcedRow kForcedB = {"M7", "G1(4)", "G1(8)", 5};
static const char* const kDataLinesB[] = {
    "  GROUP G0(2)    Z/4 n=0 len=[2, 4, 3, 3] F_2=[1, 2, 2, 2] torsion_visible=True H_Z=[Z^1 Z^1 Z^0+Z/4 Z^0+Z/2]",
    "  GROUP G0(2)    Z/4 n=1 len=[1, 3, 3, 3] F_2=[1, 3, 3, 3] torsion_visible=True H_Z=[Z^0 Z^0+Z/2 Z^0+Z/2+Z/2 Z^0+Z/2]",
    "  GROUP G0(4)    Z/4 n=0 len=[2, 5, 3, 3] F_2=[1, 3, 3, 3] torsion_visible=True H_Z=[Z^1 Z^2 Z^0+Z/2 Z^0+Z/2+Z/2]",
    "  GROUP G0(4)    Z/4 n=1 len=[1, 4, 4, 4] F_2=[1, 4, 4, 4] torsion_visible=True H_Z=[Z^0 Z^0+Z/2 Z^0+Z/2+Z/2+Z/2 Z^0+Z/2]",
    "  GROUP G0(3)    Z/9 n=0 len=[2, 3, 1, 1] F_3=[1, 2, 1, 1] torsion_visible=True H_Z=[Z^1 Z^1 Z^0+Z/6 Z^0+Z/2]",
    "  GROUP G0(9)    Z/9 n=0 len=[2, 6, 0, 0] F_3=[1, 3, 0, 0] torsion_visible=False H_Z=[Z^1 Z^3 Z^0+Z/2 Z^0+Z/2+Z/2+Z/2]",
    "  GROUP G0(5)    Z/25 n=0 len=[2, 2, 0, 0] F_5=[1, 1, 0, 0] torsion_visible=False H_Z=[Z^1 Z^1 Z^0+Z/2+Z/4 Z^0+Z/2]",
    "  GROUP G0(25)   Z/25 n=0 len=[2, 10, 0, 0] F_5=[1, 5, 0, 0] torsion_visible=False H_Z=[Z^1 Z^5 Z^0+Z/2+Z/4 Z^0+Z/2+Z/2+Z/2+Z/2+Z/2]",
    "  GROUP G1(4)    F_2 n=0 len=[1, 2, 0, 0]",
    "  GROUP G1(4)    F_3 n=1 len=[0, 2, 0, 0]",
    "  GROUP G1(8)    F_2 n=0 len=[1, 5, 0, 0]",
    "  GROUP G1(8)    F_3 n=1 len=[0, 8, 0, 0]",
    "  PAIR G0(4)<G0(2) [A:B]=2 Z/4 n=0 H(A)=[2, 4, 3, 3] H(B)=[2, 5, 3, 3] rank_res=[2, 3, 2, 2] H(A,B)=[0, 1, 3, 2]",
    "  PAIR G0(4)<G0(2) [A:B]=2 Z/4 n=1 H(A)=[1, 3, 3, 3] H(B)=[1, 4, 4, 4] rank_res=[1, 2, 2, 2] H(A,B)=[0, 1, 3, 3]",
    "  PAIR G0(9)<G0(3) [A:B]=3 Z/9 n=0 H(A)=[2, 3, 1, 1] H(B)=[2, 6, 0, 0] rank_res=[2, 2, 0, 0] H(A,B)=[0, 1, 5, 1]",
    "  PAIR G0(25)<G0(5) [A:B]=5 Z/25 n=0 H(A)=[2, 2, 0, 0] H(B)=[2, 10, 0, 0] rank_res=[2, 2, 0, 0] H(A,B)=[0, 0, 8, 0]",
    "  PAIR Gamma^2<Gamma [A:B]=2 F_2 n=0 H(A)=[1, 1, 1, 1] H(B)=[1, 1, 1, 1] rank_res=[1, 0, 1, 0] H(A,B)=[0, 1, 1, 1]",
    "  PAIR G0(4)<G0(2) [A:B]=2 F_3 n=2 H(A)=[0, 2, 0, 0] H(B)=[0, 3, 0, 0] rank_res=[0, 2, 0, 0] H(A,B)=[0, 0, 1, 0]",
    "  PAIR G0(4)<G0(2) [A:B]=2 Z/4 n=1 H(A)=[1, 3, 3, 3] H(B)=[1, 4, 4, 4] rank_res=[1, 2, 2, 2] H(A,B)=[0, 1, 3, 3]",
    "  PAIR G1(8)<G1(4) [A:B]=4 F_2 n=0 H(A)=[1, 2, 0, 0] H(B)=[1, 5, 0, 0] rank_res=[1, 1, 0, 0] H(A,B)=[0, 1, 4, 0]",
    "  PAIR G1(8)<G1(4) [A:B]=4 F_3 n=1 H(A)=[0, 2, 0, 0] H(B)=[0, 8, 0, 0] rank_res=[0, 2, 0, 0] H(A,B)=[0, 0, 6, 0]",
    "  DESC M7 G0(5) index=6 -I_in=True vertex_orbits=24 top_orbits=30 nd_S0..3=[30, 762, 15246, 318858]",
    "  DESC M7 G0(25) index=30 -I_in=True vertex_orbits=116 top_orbits=150 nd_S0..3=[150, 3810, 76230, 1594290]",
    "  DESC M7 G1(8) index=48 -I_in=False vertex_orbits=92 top_orbits=120 nd_S0..3=None",
    "  DESC R_Z Gamma^2 index=2 -I_in=True vertex_orbits=11 top_orbits=14 nd_S0..3=[14, 330, 5854, 112698]",
    "  DESC R_Z G0(2) index=3 -I_in=True vertex_orbits=15 top_orbits=21 nd_S0..3=[21, 495, 8781, 169047]",
    "  DESC R_Z G0(4) index=6 -I_in=True vertex_orbits=29 top_orbits=42 nd_S0..3=[42, 990, 17562, 338094]",
    "  FORCED_U_MINUS_I tri=M7 G1(8)->G1(4) at_least=5",
};
