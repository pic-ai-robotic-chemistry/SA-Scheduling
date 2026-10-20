# Map Asset Credits

## World Map Background

- Original work: [World map blank without borders.svg](https://commons.wikimedia.org/wiki/File:World_map_blank_without_borders.svg).
- Author: [Crates](https://commons.wikimedia.org/wiki/User:Crates); the source page records additional contributors and revisions.
- Selected license: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/), one of the licenses offered on the source page.
- Local file: `World_map_blank_without_borders.svg.webp`.
- Adaptation: a 1920 × 1068 raster representation of the SVG background, used as the base for the SDL location overlay and the Fig. 5 map panel.

The map-derived material in this directory and the Fig. 5 map panel is distributed under CC BY-SA 4.0. Reuse should preserve the author credit, source link, license link, and an indication of adaptations.

## SDL Location Reference

- Project: [The Global Self-Driving Lab Landscape](https://github.com/yetiswang/sdl-map).
- Credit: yetiswang and contributors.
- Use: reference locations for the global self-driving laboratory markers.
- Local file: `Global_SDL_118_Wikimedia_georeferenced.png`, which combines the Wikimedia background with the SDL location markers.
- Plotting adaptation: `paper_plot.py` extracts marker centers from this local reference, draws neutral site markers, and highlights five case-study locations.
- Upstream licensing statement: the [README License section](https://github.com/yetiswang/sdl-map#-license) declares the source code available under MIT.

The local image is a fixed reference used by the plotting script. For the current SDL listing and its underlying sources, consult the upstream project.

## Suggested Figure Credit

> World map adapted from “World map blank without borders.svg” by Crates, Wikimedia Commons, CC BY-SA 4.0. SDL locations based on “The Global Self-Driving Lab Landscape” by yetiswang and contributors. Marker styling and case-study highlights by the SA-Scheduling contributors. Map-derived material: CC BY-SA 4.0.

Include the source and license links above alongside this credit when sharing the map panel separately from this repository.
