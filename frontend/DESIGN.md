# Design System Specification: The Obsidian Laboratory

## 1. Overview & Creative North Star
**Creative North Star: "The Synthetic Observer"**

This design system is engineered to move beyond the "dashboard" aesthetic, evolving into a high-fidelity scientific instrument. It is rooted in **Organic Technocracy**—a philosophy that balances the cold, precise world of data with the tactile, layered depth of high-end editorial design. 

To break the "template" look, we reject the rigid, flat grid in favor of **intentional asymmetry and tonal immersion**. Key data visualizations should feel as though they are emerging from a deep obsidian void, utilizing overlapping "glass" containers and high-contrast typography scales. The layout should feel like a high-tech research environment: dense but legible, dark but vibrant, and fundamentally authoritative.

---

## 2. Colors: Tonal Depth & Radiant Accents
The palette is built on a foundation of "Obsidian" neutrals, punctuated by "Radiant Amber" for critical actions and "Royal Violet" for complex data semantics.

### The Palette (Material Design Tokens)
*   **Primary (Radiant Amber):** `#fbbc00` | **On-Primary:** `#402d00`
*   **Secondary (Royal Violet):** `#dfb7ff` | **Secondary-Container:** `#6c11af`
*   **Surface (Deep Charcoal):** `#131313`
*   **Surface-Container-Lowest:** `#0e0e0e` (The "Void" - used for the deepest background layers)
*   **Surface-Container-Highest:** `#353534` (The "Active Layer" - used for the most prominent interactive cards)

### The "No-Line" Rule
**Explicit Instruction:** Prohibit 1px solid borders for sectioning. Structural boundaries must be defined solely through background color shifts or subtle tonal transitions. 
*   *Correct:* A `surface-container-low` panel sitting on a `surface` background.
*   *Incorrect:* A grey line separating two sections of the same color.

### The "Glass & Gradient" Rule
To provide visual "soul," primary CTAs and hero data points should utilize subtle linear gradients—transitioning from `primary` (#ffe2ab) to `primary-container` (#ffbf00) at a 135-degree angle. Floating panels should employ **Glassmorphism**: use semi-transparent surface colors with a `24px` backdrop-blur to allow the "Obsidian" depth to bleed through.

---

## 3. Typography: The Monospaced Edge
The system utilizes a dual-typeface strategy to distinguish between "Control" and "Data."

*   **Display & Headlines (Space Grotesk):** Chosen for its technical, slightly wide apertures. Use `display-lg` (3.5rem) for high-impact metrics.
*   **UI Controls & Labels (Inter):** The workhorse for navigation and buttons. It provides maximum legibility at small scales (e.g., `label-sm` at 0.6875rem).
*   **The Data Signature (Monospaced Fallback):** While not in the primary scale, all numerical data and code strings must utilize a monospaced font (JetBrains Mono) to evoke a high-tech research environment.

**Hierarchy Tip:** Use extreme contrast. Pair a `display-sm` amber metric with a `label-md` muted violet description to create a sophisticated, editorial hierarchy.

---

## 4. Elevation & Depth: Tonal Layering
Traditional drop shadows are forbidden unless a component is "floating" (e.g., a modal). Depth is achieved through **Tonal Layering**.

*   **The Layering Principle:** Stack containers to create a soft, natural lift. 
    *   *Base:* `surface` (#131313)
    *   *Sub-Section:* `surface-container-low` (#1c1b1b)
    *   *Interactive Card:* `surface-container-high` (#2a2a2a)
*   **Ambient Shadows:** For floating elements, use a `24px` blur at 8% opacity. The shadow color should be tinted with `on-secondary-fixed-variant` (#690bac) to create a subtle "glow" rather than a dark smudge.
*   **The "Ghost Border":** If accessibility requires a stroke, use `outline-variant` at **15% opacity**. It should be felt, not seen.

---

## 5. Components: The Scientific Instrument

### Buttons & Controls
*   **Primary Action:** `primary-container` background with a subtle outer glow (0px 0px 8px `primary`). High-contrast `on-primary` text.
*   **Sleek Sliders:** Use a `0.25rem` (4px) track height in `surface-variant`. The thumb should be a `1rem` circle of `primary`, appearing to "float" above the track.
*   **Glow Gauges:** For visualizations, use the `secondary` purple to create "trail" effects. Use `tertiary-container` (#04dcff) as a rare tertiary accent for "Cold" data states.

### Cards & Lists
*   **The Divider Ban:** Never use horizontal lines. Separate list items using `8px` of vertical whitespace or a subtle toggle between `surface-container-low` and `surface-container-lowest`.
*   **Input Fields:** Ghost-style inputs with no bottom line. Use a `surface-container-highest` background and a `primary` indicator bar that only appears on `:focus`.

### Specialized Utility Components
*   **The Grid Overlay:** Use a repeating background pattern of dots (1px every 24px) in `outline-variant` at 5% opacity to evoke a blueprint/research feel.
*   **Data Chips:** Small-scale `secondary-container` elements with monospaced text for status tags or metadata.

---

## 6. Do’s and Don’ts

### Do
*   **Do** use asymmetrical layouts (e.g., a 70/30 split where the 30% column is a different surface tier).
*   **Do** embrace "The Void." Large areas of `surface-container-lowest` provide the luxury of focus.
*   **Do** use monospaced fonts for every single numerical value.

### Don't
*   **Don't** use pure white (#FFFFFF). Use `on-surface` (#e5e2e1) to prevent eye strain in dark environments.
*   **Don't** use standard 1px borders. They shatter the immersive, "glass-on-obsidian" feel.
*   **Don't** use rounded corners larger than `0.75rem`. This system is professional and precise; overly round "bubble" shapes undermine its scientific authority.

### Accessibility Note
Ensure that all `primary` amber text on `surface` backgrounds maintains a contrast ratio of at least 4.5:1. Use the `on-primary-container` (#6d5000) for small text if the base amber is too bright for long-form reading.