## Why this project

I use AI tools all the time, but I didn't really understand how a neural network learns. I wanted to build one myself and see how it goes from random guesses to useful predictions. I chose exoplanets because I've always been curious about space, and finding planets around other stars is a real problem scientists are still working on. It let me learn machine learning on real NASA data instead of a practice dataset, and learn some astronomy along the way.

## Nov 20, Step 2
- KOI table: 2,748 confirmed, 4,839 false positives, 1,977 candidates (dropped)
- Picked a balanced 400 + 400 so the model can't just guess "fake"
- Found missing values almost only in false positives, so I dropped them before
  sampling; otherwise the model could learn "missing = fake" as a shortcut