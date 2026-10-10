# HiDreamGenerator design system

The accepted desktop references are `design/create-concept.png` and
`design/edit-concept.png` at 1536×1024. They define one application shell with
two workflow states.

## Tokens

- Background: warm paper `#f4f1ea`; raised control fill `#fbfaf7`.
- Text: ink `#171717`; secondary graphite `#66635d`.
- Rules: `#cfcbc2`; strong rules `#aaa59b`.
- Accent: ultramarine `#1757ff`; live dot: `#f06b17`; error: `#b42318`.
- Geometry: 64px top bar, 360px control rail, 8px controls, thin borders,
  almost no elevation.
- UI type: system grotesk at 12–16px. Stage heading: Georgia/display serif.

## Component inventory

- Quiet top bar with the H mark, product name, local GPU status, help/settings.
- Create/Edit tab pair with underline selection.
- Reusable field labels, textarea, secondary text action, preset buttons,
  number input, switch, and full-width primary action.
- Edit-only dropzone/source thumbnail, original-aspect switch, and two-option
  edit-method selector.
- Open workspace stage for empty, source, progress-preview, result, and error.
- Bottom session rail for generated outputs, selection, clearing, and download.

## Responsive model

At 900px and below the rail becomes a normal top section and the workspace
continues below it. At phone widths, preset and method controls remain tappable,
the top-bar utility copy condenses, and the session rail scrolls horizontally.

The H mark is intentionally code-native because it is a simple interface glyph
whose exact geometry is better preserved as text/CSS than as a raster asset.
