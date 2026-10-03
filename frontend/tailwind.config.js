/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        forensic: {
          'light-background': '#f8fafc',
          'light-surface': '#ffffff',
          'light-text': '#0f172a',
          'light-muted': '#475569',
          'light-border': '#cbd5e1',
          'light-primary': '#0f766e',
          'dark-background': '#020617',
          'dark-surface': '#0f172a',
          'dark-text': '#cbd5e1',
          'dark-muted': '#94a3b8',
          'dark-border': '#334155',
          'dark-primary': '#22d3ee',
        },
      },
    },
  },
}