Gap Status Real cost to close
1D multi-frequency additive (cos(2t)+0.5·sin(0.5t)) Fails; greedy MRE absorbs both into one fit ~2–3 hours: BIC-based hypothesis test before composition
Real measurement data (Keeling attempt) Failed; trend masks seasonal, no candidates survive MDL Days: detrending must be a first-class stage in the loop
Implicit laws (Nikuradse-style f(x,y)=0) Structurally out of scope Weeks: fixed-point solver as a construct
nD beyond 3D Untested No new code, just runtime; n=5 will be ~10× slower
MDL structural-vs-residual weighting Mitigated by gates, not fixed at scorer level 1–2 days: real MDL rewrite
"Design new real unknown experiments" Not demonstrated The loop picks measurement points from a known oracle. Real autonomy against unknown physics hasn't been shown