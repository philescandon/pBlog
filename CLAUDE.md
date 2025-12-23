# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal blog built with Hugo and R's blogdown package. The site uses the hugo-xminBlogTemplate theme (a minimal theme by Yihui Xie) and is deployed via Netlify.

## Build Commands

```bash
# Local development - serve the site with live reload
hugo server

# Build for production
hugo

# Using blogdown in R (alternative workflow)
blogdown::serve_site()    # Start local server
blogdown::build_site()    # Build static files
```

Hugo version: 0.91.0 (pinned in both `.Rprofile` and `netlify.toml`)

## Architecture

### Content Structure
- `content/post/` - Blog posts (full articles with categories/tags)
- `content/note/` - Short notes
- `content/about.md` - About page

Posts use YAML frontmatter with: `title`, `author`, `date`, `slug`, `categories`, `tags`

### R Markdown Support
- `.Rmd` and `.Rmarkdown` files are automatically knit to HTML/Markdown on save (configured in `.Rprofile`)
- `blogdown.method = 'html'` means Rmd files render via Pandoc to HTML
- MathJax is enabled via `layouts/partials/foot_custom.html`

### Theme Customization
- Base theme: `themes/hugo-xminBlogTemplate/`
- Local overrides: `layouts/` directory (takes precedence over theme layouts)
- Custom footer scripts in `layouts/partials/foot_custom.html` (MathJax, image centering)

### Build Hooks
- `R/build.R` - Runs before Hugo builds (currently empty)
- `R/build2.R` - Runs after Hugo builds (currently empty)

## Deployment

Netlify automatically builds on push to main. Configuration in `netlify.toml`:
- Production: `hugo` with `HUGO_ENV=production`
- Branch/preview deploys: `hugo -F -b $DEPLOY_PRIME_URL`
