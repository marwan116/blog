const withNextra = require('nextra')({
    theme: 'nextra-theme-blog',
    themeConfig: './theme.config.js',
    latex: true
})
module.exports = withNextra()
