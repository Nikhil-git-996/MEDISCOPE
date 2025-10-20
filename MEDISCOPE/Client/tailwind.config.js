// tailwind.config.js
module.exports = {
  darkMode: 'class', // This line enables dark mode based on a 'dark' class in your HTML
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        poppins: ['Poppins', 'sans-serif'], // Define Poppins font
      },
    },
  },
  plugins: [],
};