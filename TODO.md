# TODO

## Decided

- [ ] **Switch covariance from n to n−1.** The app currently divides by n (population covariance), so its distances are about 2.6% larger than NumPy's and SciPy's with 20 points. Use n−1 (sample covariance), the standard formula for data that is a sample. The formula lives in `calculateMeanAndCovariance()` in `src/utils/math.ts`; update the hand-computed values in `src/utils/math.test.ts` with it. Add a short note in the UI saying which formula is used.
  - Later (not now): add a dropdown to switch between n−1 and n.
- [ ] **Show how much data each ellipse covers.** In 2D, the 1-std, 2-std and 3-std ellipses hold about 39%, 86% and 99% of Gaussian data, not the 68%, 95% and 99.7% people remember from the 1D bell curve. Add these percentages to the ellipse legend labels (for example, "1-std ellipse (~39%)") so the demo doesn't suggest the 1D numbers.
- [ ] **Generate correlated random data.** Dataset points are currently uniform, with x and y both in [-5, 25], so the cloud is nearly a circle and Mahalanobis distance looks almost the same as Euclidean distance. Generate dataset points from a tilted, stretched Gaussian with a random angle and stretch on each regenerate, and keep test points spread over a wider area so some fall inside and some outside the ellipses.
- [ ] **Add component tests.** `src/utils/math.test.ts` covers the math against hand-computed values, and `src/App.test.tsx` is a render smoke test with the chart stubbed out, because jsdom has no canvas. Add tests for the buttons and table edits (add, regenerate, edit, delete).

## Suggested (not yet decided)

- [ ] **Create the initial random points in `useState` instead of `useEffect`.** `MahalanobisPlot.tsx` renders once with empty data, then fills it in from an effect. oxlint warns about this (`react(set-state-in-effect)`). Passing an initializer function to `useState` avoids the extra render.
- [ ] **Reduce the bundle size.** The production bundle is about 1.6 MB (516 kB gzipped), mostly the full ECharts and mathjs libraries, and Vite warns about it. Importing only the ECharts parts the chart uses (`echarts/core` plus the scatter and line charts) would shrink it a lot. mathjs is now used only for `inv()` on the 2×2 covariance matrix in `calculateMahalanobisDistance()`; writing that inverse out by hand would remove the dependency.
- [ ] **Place a new dataset point away from a single existing one.** With exactly one dataset point, "Add Dataset Point" uses that point's range, which is zero, so the new point lands on top of it. The two identical points give a near-zero covariance, tiny ellipses, and test-point distances in the tens of thousands. `addRandomPoint()` in `MahalanobisPlot.tsx` needs a minimum range, as it already has for the empty case.
- [ ] **Hide the ellipse legend entries when there are no ellipses.** With fewer than 2 dataset points, the legend still lists the three ellipses, and ECharts logs "series not exists" warnings for each one.
- [ ] **Replace the default icons.** `public/favicon.ico`, `logo192.png` and `logo512.png` are still the React logos from the Create React App template.
