# Sample data

Original, simple mechanical parts (no licensing issues) so you can try the agent right away.
Regenerate everything with `python -m scripts.make_samples --force`.

## CAD files (`cad/`)

| File | Design | Format |
|---|---|---|
| `plate_2holes.step` | flat plate, 2 holes | STEP |
| `plate_3holes.step` | same plate, 3 holes (same size, different design) | STEP |
| `flange_50mm.step` | round flange, center bore + 4 bolt holes | STEP |
| `flange_500mm.step` | the same flange, **10x larger** | STEP |
| `l_bracket.stl` | L-bracket with one hole | STL |
| `t_bracket.obj` | T-shaped bracket | OBJ |
| `hex_nut.ply` | hex nut | PLY |
| `spacer_tube.stl` | tube spacer | STL |
| `sprocket.stp` | 10-tooth sprocket with bore | STP |

## Photos (`photos/`)

Synthetic photos: each part rendered with a perspective camera, metallic color, random
viewpoint and rotation, a textured table-like background, blur, noise and JPEG compression.
`labels.csv` records which design each photo shows.

## Try these pairs

| What it shows | CAD (reference) | Photo | Expected |
|---|---|---|---|
| Matching pair | `plate_2holes.step` | `plate_2holes_photo.jpg` | MATCH |
| Scaled matching pair (object is 10x the CAD) | `flange_50mm.step` | `flange_500mm_photo.jpg` | MATCH |
| Same size, different design (3 holes vs 2) | `plate_2holes.step` | `plate_3holes_photo.jpg` | NOT A MATCH |
| Completely different design | `sprocket.stp` | `l_bracket_photo.jpg` | NOT A MATCH |
| Library search | (whole library) | `t_bracket_photo.jpg` | MATCH: t_bracket |
