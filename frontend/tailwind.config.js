/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#111827', // Gray 900
          800: '#1f2937', // Gray 800
          700: '#374151', // Gray 700
          400: '#9ca3af', // Gray 400
        },
        primary: {
          500: '#3b82f6', // Blue 500 (Enterprise blue)
          600: '#2563eb', // Blue 600
        },
        danger: '#ef4444', // Red for unsafe
        success: '#10b981', // Green for safe
      }
    },
  },
  plugins: [],
}
