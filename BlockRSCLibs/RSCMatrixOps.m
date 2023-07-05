(* ::Package:: *)

(*This package defines operations on matrices which are the form RSC (Replica-Symmetric or Constant), which generalized RS (Replica-Symmetric) matrices*)
(*Matrices are defined by a list of 4 numbers {N, M, a, b}.  Then A = a I_N \delta_{N,M} + b 11^t*)
(*For Example, {N, N, 1, 4} represents an N by N matrix of the form I + 4 11^t *)
(* {N, M, 0, 2} represents an N by M matrix of the form 2 1_N 1_M^t*)
(* If N neq M, we require  a == 0*)
BeginPackage["RSCMatrixOps`"]

RSCadd::usage = "Adds two RSC matrices provided they have compatible indices."
RSCsubtract::usage = "Subtracts two RSC matrices provided they have compatible indices."
RSCaddlist::usage = "Adds a list of RSC matrices provided they have compatible indices."
RSCmultiply::usage = "Multiplies two RSC matrices provided they have compatible indices"
RSCmultiplylist::usage = "Multiplies a list of RSC matrices provided they have compatible indices"
RSCinverse::usage = "Inverse of square RS matrix"
RSCtrace::usage = "Trace of square RS matrix"
RSCsum::usage = "Sum over all elements of an RSC matrix"
RSCscalarmultiply::usage = "scalar multiple of an RSC matrix"
RSretile::usage = "Assigns a new tiling to an RS matrix (square) with the given dimensions of the diagonal blocks.  Returns a BlockRSCMatrix"
RSCtranspose::usage = "Takes the transpose of an RSC matrix."

Begin["`Private`"] (* Begin Private Context *)

RSCadd[A_, B_] := (
	Assert[Simplify[A[[1]]]==Simplify[B[[1]]] && Simplify[A[[2]]]==Simplify[B[[2]]], "Added Matrices have Incompatible Dimensions"];
	{A[[1]], A[[2]], A[[3]]+B[[3]], A[[4]]+B[[4]]}
)

RSCsubtract[A_, B_] := (
	Assert[Simplify[A[[1]]]==Simplify[B[[1]]] && Simplify[A[[2]]]==Simplify[B[[2]]], "Added Matrices have Incompatible Dimensions"];
	{A[[1]], A[[2]], A[[3]]-B[[3]], A[[4]]-B[[4]]}
)

RSCaddlist[list_] := Module[{},
  If[
    AllTrue[Rest[list], #1[[1]] == list[[1, 1]] &] && 
    AllTrue[Rest[list], #1[[2]] == list[[1, 2]] &],
    Fold[( {#1[[1]], #1[[2]], #1[[3]] + #2[[3]], #1[[4]] + #2[[4]]} &), list],
    Print["Error!!!!  The dimensions of the added matrices do not all match!"];
    list
  ]
]

RSCmultiply[A_, B_] := (
	Assert[Simplify[A[[2]]]== Simplify[B[[1]]], "Multiplied Matrices have Incompatible Dimensions"];
	{A[[1]], B[[2]], A[[3]]*B[[3]], A[[3]]*B[[4]]+A[[4]]*B[[3]]+A[[2]]*A[[4]]*B[[4]]}
)

RSCmultiplylist[list_] := Module[{pairs},
  pairs = Transpose[{list, RotateLeft[list]}];
  If[AllTrue[Most[pairs], #1[[1, 2]] == #1[[2, 1]] &],
    Fold[( {#1[[1]], #2[[2]], #1[[3]]*#2[[3]], #1[[3]]*#2[[4]]+ #1[[4]]*#2[[3]]+ #1[[2]]*#1[[4]]*#2[[4]]} &), list],
    Print["The dimensions of multiplied matrices are incompatible!"];
    list
  ]
]

RSCinverse[{N_, M_, a_, b_}] := Module[{x, y},
  Assert[Simplify[N] == Simplify[M], "Inverse is only defined for square matrices"];
  x = 1 / a;
  y = -b / (a^2 + N * a * b);
  {N, M, x, y}
]

RSCtrace[{N_, M_, a_, b_}] := Module[{},
    Assert[Simplify[N] == Simplify[M], "Trace should only be used on square matrices"];
    N*(a+b)
]

RSCsum[{N_, M_, a_, b_}] := N*a + N*M*b

RSCscalarmultiply[c_, M_] := (
	Assert[Length[M] == 4, "RSC Matrix must be quadruple"];
	{M[[1]], M[[2]], c*M[[3]], c*M[[4]]}
)

RSretile[M_, dims_] := Module[{k, i, j, result},
    k = Length[dims];
    Assert[Simplify[M[[1]]]== Simplify[M[[2]]], "Input matrix must be square"];
    Assert[Simplify[Total[dims]] == Simplify[M[[1]]], "Re-Tiling must have same total dimension as the input."];
    result = Table[0, {k}, {k}];
    For[i = 1, i <= k, i++,
        For[j = 1, j <= k, j++,
            result[[i,j]] = If[i==j, {dims[[i]], dims[[j]],M[[3]],M[[4]]}, {dims[[i]], dims[[j]],0,M[[4]]}];
        ]
    ];
    result
]

RSCtranspose[M_] := {M[[2]], M[[1]], M[[3]], M[[4]]}

End[] (* End Private Context *)

EndPackage[]

