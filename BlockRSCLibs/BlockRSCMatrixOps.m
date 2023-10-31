(* ::Package:: *)

BeginPackage["BlockRSCMatrixOps`", {"RSCMatrixOps`"}]

BlockRSCadd::usage = "Adds two BlockRSC matrices provided they have compatible tilings."
BlockRSCsubtract::usage = "Subtracts two BlockRSC matrices provided they have compatible tilings."
BlockRSCaddlist::usage = "Adds a list of BlockRSC matrices provided they have compatible tilings."
BlockRSCmultiply::usage = "Multiplies two BlockRSC matrices provided they have compatible tilings"
BlockRSCmultiplylist::usage = "Multiplies a list of BlockRSC matrices provided they have compatible tilings"
BlockRSCinverse::usage = "Inverts a square BlockRSC Matrix recursively using block matrix inversion at each step."
BlockRSCtrace::usage = "Trace of square BlockRS matrix"
BlockRSCsum::usage = "Sums all elements of a BlockRS matrix"
BlockRSCscalarmultiply::usage = "multiply BlockRSC matrix by a scalar."
BlockRSCtranspose::usage = "Takes the transpose of a BlockRSC matrix."
BlockRSCinverse::usage = "Recursive algorithm to invert BlockRSC matrix"
BlockRSCdet::usage = "Recursive algorithm to take determinant of BlockRSC matrix"

Begin["`Private`"] (* Begin Private Context *)

BlockRSCadd[M1_, M2_] := MapThread[RSCMatrixOps`RSCadd, {M1, M2}, 2]
BlockRSCsubtract[M1_, M2_] := MapThread[RSCMatrixOps`RSCsubtract, {M1, M2}, 2]

BlockRSCaddlist[list_] := Fold[(BlockRSCadd[#1, #2] &), list]

BlockRSCmultiply[A_, B_] := Module[{n, m, p, x, i, j, k, result},
    {n, m, x} = Dimensions[A];
    {m, p, x} = Dimensions[B];

    Assert[Simplify[m] == Simplify[Dimensions[B][[1]]], "The inner matrix dimensions must agree."];

    result = Table[0, {n}, {p}];
    For[i = 1, i <= n, i++,
        For[j = 1, j <= p, j++,
            result[[i,j]] = {A[[i,1,1]], B[[1,j,2]],0,0};
            For[k = 1, k <= m, k++,
                result[[i, j]] = RSCMatrixOps`RSCadd[result[[i, j]], RSCMatrixOps`RSCmultiply[A[[i, k]], B[[k, j]]]]
            ]
        ]
    ];
    result
]

BlockRSCmultiplylist[list_] := Fold[(BlockRSCmultiply[#1, #2] &), list]

BlockRSCscalarmultiply[c_, A_] := Map[RSCMatrixOps`RSCscalarmultiply[c, #] &, A, {2}]

BlockRSCtrace[A_] := Module[{n, m, x, i, result},
    {n, m, x} = Dimensions[A];
    Assert[Simplify[n] == Simplify[m], "Matrix must be square"];
    result = 0;
    For[i = 1, i <= n, i++,
        result = result + RSCMatrixOps`RSCtrace[A[[i, i]]];
    ];
    result
]

BlockRSCsum[A_] := Module[{n, m, x, i, j, result},
    {n, m, x} = Dimensions[A];
    result = 0;
    For[i = 1, i <= n, i++,
        For[j = 1, j <= m, j++,
            result = result + RSCMatrixOps`RSCsum[A[[i, j]]];
        ]
    ];
    result
]

BlockRSCtranspose[A_] := Map[RSCMatrixOps`RSCtranspose[#] &, Transpose[A], {2}]

BlockRSCinverse[A_] := Module[{UL, UR, BL, BR, ULinv, SchurComplement, SchurComplementinv, newUL, newUR, newBL, newBR, n, m, x, result},
    {n, m, x} = Dimensions[A];

    Assert[Simplify[n] == Simplify[m], "Matrix must be square"];
    If[n == 1,
        result = {{RSCinverse[A[[1, 1]]]}},
        UL = A[[1;;1, 1;;1]];
        UR = A[[1;;1, 2;;m]];
        BL = A[[2;;n, 1;;1]];
        BR = A[[2;;n, 2;;m]];

        ULinv = {{RSCinverse[UL[[1,1]]]}};(*UL is a single RS matrix*)

        SchurComplement = BlockRSCsubtract[BR, BlockRSCmultiply[BlockRSCmultiply[BL, ULinv], UR]]; (* Assuming you have BlockRSCsubtract function *)
        SchurComplementinv = BlockRSCinverse[SchurComplement]; (* Recursive call here! *)

        newUL = BlockRSCadd[ULinv, BlockRSCmultiplylist[{ULinv, UR, SchurComplementinv, BL, ULinv}]];
        newUR = BlockRSCscalarmultiply[-1, BlockRSCmultiplylist[{ULinv, UR, SchurComplementinv}]];
        newBL = BlockRSCscalarmultiply[-1, BlockRSCmultiplylist[{SchurComplementinv, BL, ULinv}]];
        newBR = SchurComplementinv;
        result = Join[
            Join[Simplify[newUL], Simplify[newUR], 2],
            Join[Simplify[newBL], Simplify[newBR], 2]
        ];
    ];
    result
]
BlockRSCdet[A_] := Module[{UL, UR, BL, BR, ULinv, ULdet, SchurComplement, SchurComplementdet, n, m, x, result},
    {n, m, x} = Dimensions[A];

    Assert[Simplify[n] == Simplify[m], "Matrix must be square"];
    If[n == 1,
        result = RSCdet[A[[1, 1]]],
        UL = A[[1;;1, 1;;1]];
        UR = A[[1;;1, 2;;m]];
        BL = A[[2;;n, 1;;1]];
        BR = A[[2;;n, 2;;m]];

        ULinv = {{RSCinverse[UL[[1,1]]]}};(*UL is a single RS matrix*)
        ULdet = RSCdet[UL[[1,1]]];(*UL is a single RS matrix*)

        SchurComplement = BlockRSCsubtract[BR, BlockRSCmultiplylist[{BL, ULinv, UR}]]; (* Assuming you have BlockRSCsubtract function *)
        SchurComplementdet = BlockRSCdet[SchurComplement]; (* Recursive call here! *)
        
        result = ULdet*SchurComplementdet;
    ];
    result
]

End[] (* End Private Context *)

EndPackage[]



