## Why this project

I use AI tools all the time, but I didn't really understand how a neural network learns. I wanted to build one myself and see how it goes from random guesses to useful predictions. I chose exoplanets because I've always been curious about space, and finding planets around other stars is a real problem scientists are still working on. It let me learn machine learning on real NASA data instead of a practice dataset, and learn some astronomy along the way.

## Nov 20, Step 2
- KOI table: 2,748 confirmed, 4,839 false positives, 1,977 candidates (dropped)
- Picked a balanced 400 + 400 so the model can't just guess "fake"
- Found missing values almost only in false positives, so I dropped them before
  sampling; otherwise the model could learn "missing = fake" as a shortcut

## Nov 22, Step 3
- Processed 796/800 signals into two 201-point views: global (whole orbit) and local (zoomed on the dip)
- Added the local view because in the global view a short dip was only ~5 points wide, too small for the CNN to see its shape
- Bug: 13 fakes crashed because their "dip" was a third of the orbit, so hiding the dips hid everything. Capped the mask at half the orbit instead of dropping them
- Check: the cleaned dip for KOI-923.01 was ~1,250 ppm vs NASA's 1,240
- Flagged 17 signals whose local view was mostly interpolated; all 17 are false positives
  (incl. the 13 with dips ≥ 1/3 of the orbit). Keeping them, but will check CNN results with/without them


- Guess: simple model recall on Kepler test set ≈ 75%

## Nov 23, Step 4: simple model (written myself)
- Logistic regression from scratch: log10 + standardise, sigmoid, cross-entropy, gradient descent
- Loss 0.693 → 0.634. Test: accuracy 0.78, precision 0.73, recall 0.89
- Weights: depth −0.53, duration −0.51, period +0.64 (deep/long dips → fake)
- Mistakes are mostly false alarms (26 FP vs 9 FN): ~1/3 of fakes look like planets in these 3 numbers, same as my EDA showed. A straight-line model can't catch fakes at both extremes. The CNN needs the dip shape