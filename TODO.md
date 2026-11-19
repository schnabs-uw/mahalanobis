# TODO

## Decided

- [ ] **Remove the duplicated covariance code.** The mean and covariance math is copied in three places: `getChartOptions()` in `src/components/MahalanobisPlot.tsx`, and `calculateMahalanobisDistance()` and `calculateEllipse()` in `src/utils/math.ts`. Move it into one helper in `math.ts` (for example, `calculateMeanAndCovariance(points)`), compute it once per render, and pass the result to the distance, ellipse and eigenvector functions.
- [ ] **Switch covariance from n to n−1.** The app currently divides by n (population covariance), so its distances are about 2.6% larger than NumPy's and SciPy's with 20 points. Use n−1 (sample covariance), the standard formula for data that is a sample. Add a short note in the UI saying which formula is used.
  - Later (not now): add a dropdown to switch between n−1 and n.
- [ ] **Show how much data each ellipse covers.** In 2D, the 1-std, 2-std and 3-std ellipses hold about 39%, 86% and 99% of Gaussian data, not the 68%, 95% and 99.7% people remember from the 1D bell curve. Add these percentages to the ellipse legend labels (for example, "1-std ellipse (~39%)") so the demo doesn't suggest the 1D numbers.
- [ ] **Generate correlated random data.** Dataset points are currently uniform, with x and y both in [-5, 25], so the cloud is nearly a circle and Mahalanobis distance looks almost the same as Euclidean distance. Generate dataset points from a tilted, stretched Gaussian with a random angle and stretch on each regenerate, and keep test points spread over a wider area so some fall inside and some outside the ellipses.
- [ ] **Replace power iteration with the exact 2×2 eigen solution.** `calculateEigenvectors()` in `src/utils/math.ts` runs 100 power-iteration steps. It is accurate in most runs but occasionally wrong for nearly circular clouds: in a simulation of 2,000 datasets like the app's current ones, about 0.6% of runs were off by more than 5°, and the worst was off by 52°. When that happens the ellipses and arrows are drawn rotated wrong, while the distance labels stay correct. Use the closed-form solution for symmetric 2×2 matrices and keep the existing sign rule (`normalizeDirection`) so the arrows don't flip.
- [ ] **Add more tests.** Today there is one smoke test (`src/App.test.tsx`) that renders the app with the chart stubbed out, because jsdom has no canvas. Start with unit tests for `src/utils/math.ts` (distance, eigenvalues and eigenvectors, ellipse points), checked against known values. Then add component tests for the buttons and table edits.

## Suggested (not yet decided)

- [ ] **Create the initial random points in `useState` instead of `useEffect`.** `MahalanobisPlot.tsx` renders once with empty data, then fills it in from an effect. oxlint warns about this (`react(set-state-in-effect)`). Passing an initializer function to `useState` avoids the extra render.
- [ ] **Reduce the bundle size.** The production bundle is about 1.6 MB (516 kB gzipped), mostly the full ECharts and mathjs libraries, and Vite warns about it. Importing only the ECharts parts the chart uses (`echarts/core` plus the scatter and line charts) would shrink it a lot. Once the eigen solution is closed-form, mathjs may not be needed at all.
- [ ] **Replace the default icons.** `public/favicon.ico`, `logo192.png` and `logo512.png` are still the React logos from the Create React App template.
