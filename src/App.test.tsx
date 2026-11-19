import { render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import App from './App'

// jsdom has no canvas, so ECharts cannot render in tests.
vi.mock('echarts-for-react', () => ({
  default: () => <div data-testid="chart" />,
}))

test('renders the controls, chart and point tables', () => {
  render(<App />)
  expect(screen.getByRole('button', { name: 'Add Dataset Point' })).toBeInTheDocument()
  expect(screen.getByTestId('chart')).toBeInTheDocument()
  expect(screen.getByText('Dataset Points')).toBeInTheDocument()
  expect(screen.getByText('Test Points')).toBeInTheDocument()
})
