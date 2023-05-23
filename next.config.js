const withNextra = require('nextra')('nextra-theme-blog', './theme.config.js')

const theme = require("shiki/themes/nord.json")
const {
    remarkCodeHike,
} = require("@code-hike/mdx")

const withMDX = require("@next/mdx")({
    extension: /\.mdx?$/,
    options: {
        remarkPlugins: [
            [remarkCodeHike, { theme }]
        ],
    },
})

module.exports = [
    withNextra(),
    withMDX({
        pageExtensions: [
            "ts", "tsx", "js",
            "jsx", "md", "mdx"
        ],
    })
]