# Mahalanobis Distance Interactive Demo

An interactive, browser-only visualization of [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance): how far a point is from a cloud of points once the cloud's spread and correlation are taken into account.

Live demo: https://schnabs-uw.github.io/mahalanobis/

## What it shows

- A random dataset (blue) and random test points (red), each test point labeled with its Mahalanobis distance from the dataset
- The 1, 2 and 3 standard-deviation covariance ellipses of the dataset
- The covariance matrix's eigenvectors (the ellipse axes), labeled with their eigenvalues
- Buttons to add points and to regenerate all points, only the dataset, or only the test points
- Editable tables to change or delete individual points; the plot updates immediately

All calculations run in the browser.

## Tech stack

- [React](https://react.dev/) 19 and TypeScript
- [Vite](https://vite.dev/) for the dev server and build
- [MUI](https://mui.com/) for the UI components
- [ECharts](https://echarts.apache.org/) (via `echarts-for-react`) for the chart
- [mathjs](https://mathjs.org/) for matrix operations
- [Vitest](https://vitest.dev/) and React Testing Library for tests, [oxlint](https://oxc.rs/docs/guide/usage/linter) for linting

## Getting started

Requires Node.js 20.19+ or 22.12+ (CI uses Node 24) and npm.

```sh
npm install
npm run dev
```

Then open http://localhost:5173/mahalanobis/.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Start the dev server with hot reload |
| `npm run build` | Type-check and build the production site into `dist/` |
| `npm run preview` | Serve the built `dist/` locally |
| `npm run typecheck` | Type-check only |
| `npm run lint` | Lint with oxlint |
| `npm test` | Run the tests once |
| `npm run test:watch` | Run the tests in watch mode |

## Project layout

```
src/
├── main.tsx                      Entry point
├── App.tsx                       MUI theme and page layout
├── components/MahalanobisPlot.tsx  Controls, chart and point tables
└── utils/math.ts                 Distance, eigenvector and ellipse math
```

## Deployment

`.github/workflows/deploy.yml` lints, tests and builds every pull request and push. Pushes to `master` also publish `dist/` to GitHub Pages. The site is served from `/mahalanobis/`, which is set as `base` in `vite.config.ts`.

## License

MIT
