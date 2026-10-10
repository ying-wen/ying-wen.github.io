/**
 * Shared, dependency-free drawing vocabulary for the CRL textbook.
 * Helpers render supplied quantities; they never infer a transition, value,
 * policy, uncertainty interval, or empirical result from visual placement.
 */
export const palette = Object.freeze({
  ink: '#19354e', muted: '#52687b', line: '#c8d6e0', blue: '#146ca1',
  teal: '#258375', orange: '#b65a28', purple: '#7854a2', red: '#b54142',
  white: '#ffffff', bg: '#f5f8fb',
});
export const typography = Object.freeze({
  family: 'system-ui,-apple-system,"PingFang SC","Noto Sans CJK SC",sans-serif',
  title: 26, label: 20, secondary: 18, tick: 16,
});
export const spacing = Object.freeze({outer: 28, gap: 24, contentTop: 80});
export const escapeXML = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&apos;');
const finite = (value, name) => {if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite`); return value;};
const positive = (value, name) => {finite(value, name); if (value <= 0) throw new RangeError(`${name} must be positive`); return value;};
const number = value => Number(finite(value, 'coordinate').toFixed(4));
const attr = escapeXML;
const dashAttr = dash => dash ? ` stroke-dasharray="${attr(Array.isArray(dash) ? dash.join(' ') : dash === true ? '8 6' : dash)}"` : '';

export function text(x, y, value, {size = 20, color = palette.ink, anchor = 'start', weight = 400, opacity = 1} = {}) {
  positive(size, 'font size');
  if (!['start', 'middle', 'end'].includes(anchor)) throw new RangeError('text anchor must be start, middle, or end');
  return `<text x="${number(x)}" y="${number(y)}" font-size="${size}" fill="${attr(color)}" text-anchor="${anchor}" font-weight="${weight}" opacity="${opacity}">${escapeXML(value)}</text>`;
}
export function line(x1, y1, x2, y2, {color = palette.line, width = 2, dash = '', opacity = 1} = {}) {
  positive(width, 'stroke width');
  return `<path d="M${number(x1)} ${number(y1)}L${number(x2)} ${number(y2)}" fill="none" stroke="${attr(color)}" stroke-width="${width}" stroke-linecap="round" opacity="${opacity}"${dashAttr(dash)}/>`;
}
export function arrow(x1, y1, x2, y2, {mode = 'actual', color = mode === 'imagined' ? palette.purple : palette.blue, width = 3, dash = mode === 'imagined' ? '8 6' : '', head = 10, label, labelOffset = -12, ...rest} = {}) {
  if (!['actual', 'imagined'].includes(mode)) throw new RangeError('arrow mode must be actual or imagined');
  const length = Math.hypot(x2 - x1, y2 - y1);
  if (!length) throw new RangeError('an arrow needs distinct endpoints');
  positive(head, 'arrow head');
  const h = Math.min(head, length / 2), ux = (x2 - x1) / length, uy = (y2 - y1) / length;
  // End the shaft under a solid triangle. The head remains solid in dashed mode.
  const bx = x2 - h * ux, by = y2 - h * uy, half = h * .5;
  let out = line(x1, y1, x2 - h * .65 * ux, y2 - h * .65 * uy, {color, width, dash, ...rest});
  out += `<path data-arrow-mode="${mode}" d="M${number(x2)} ${number(y2)}L${number(bx - uy * half)} ${number(by + ux * half)}L${number(bx + uy * half)} ${number(by - ux * half)}Z" fill="${attr(color)}"/>`;
  if (label !== undefined) out += text((x1 + x2) / 2, (y1 + y2) / 2 + labelOffset, label, {size: typography.secondary, color, anchor: 'middle'});
  return out;
}
export function rect(x, y, width, height, {fill = palette.white, stroke = palette.line, radius = 6, strokeWidth = 1.5, opacity = 1} = {}) {
  if (width < 0 || height < 0) throw new RangeError('rectangle dimensions cannot be negative');
  return `<rect x="${number(x)}" y="${number(y)}" width="${number(width)}" height="${number(height)}" rx="${number(radius)}" fill="${attr(fill)}" stroke="${attr(stroke)}" stroke-width="${strokeWidth}" opacity="${opacity}"/>`;
}
export function circle(x, y, radius, {fill = palette.blue, stroke = 'none', strokeWidth = 2, opacity = 1} = {}) {
  if (radius < 0) throw new RangeError('circle radius cannot be negative');
  return `<circle cx="${number(x)}" cy="${number(y)}" r="${number(radius)}" fill="${attr(fill)}" stroke="${attr(stroke)}" stroke-width="${strokeWidth}" opacity="${opacity}"/>`;
}
export function agent(x, y, {radius = 12, color = palette.blue, heading, label} = {}) {
  let out = circle(x, y, radius, {fill: color, stroke: palette.white, strokeWidth: 2.5});
  if (heading !== undefined) {
    finite(heading, 'heading'); // Radians clockwise in SVG coordinates: 0 points right.
    out += line(x, y, x + .65 * radius * Math.cos(heading), y + .65 * radius * Math.sin(heading), {color: palette.white, width: 2.5});
  }
  if (label !== undefined) out += text(x, y + radius + 24, label, {anchor: 'middle', size: 18});
  return out;
}
export function panel(x, y, width, height, {title, body = '', fill = palette.bg, stroke = 'none', labelSize = 20} = {}) {
  return rect(x, y, width, height, {fill, stroke, radius: 8}) + (title ? text(x + 16, y + 28, title, {size: labelSize}) : '') + body;
}

const hash = value => {let h = 2166136261; for (const c of value) h = Math.imul(h ^ c.codePointAt(0), 16777619); return (h >>> 0).toString(36);};
export function frame({title, description, width = 720, height, body, kind = 'schematic', heading = {}, background = palette.white, defs = '', idPrefix} = {}) {
  if (!title || !description) throw new TypeError('figure title and description are required');
  positive(width, 'figure width'); positive(height, 'figure height');
  if (!['schematic', 'exact', 'measured'].includes(kind)) throw new RangeError('kind must be schematic, exact, or measured');
  const id = idPrefix ?? `crl-${hash(`${title}|${description}|${width}|${height}`)}`;
  if (!/^[A-Za-z][A-Za-z0-9_.-]*$/.test(id)) throw new TypeError('idPrefix must be an XML-safe identifier');
  const headingSVG = heading === false ? '' : text(heading.x ?? 28, heading.y ?? 42, title, {size: heading.size ?? typography.title, color: heading.color ?? palette.ink});
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${id}-title ${id}-desc" data-crl-visual="v1" data-evidence-kind="${kind}"><title id="${id}-title">${escapeXML(title)}</title><desc id="${id}-desc">${escapeXML(description)}</desc>${defs ? `<defs>${defs}</defs>` : ''}<rect width="${width}" height="${height}" fill="${attr(background)}"/><g font-family="${attr(typography.family)}">${headingSVG}${body ?? ''}</g></svg>\n`;
}

/**
 * Explicit row/column geometry. Cell coordinates are [column, row].
 * `observed` masks the omitted cells; it never draws hidden walls/values below
 * translucent fog. For an omniscient + local-view comparison draw two worlds.
 * A terminal is an endpoint (double border), not automatically a positive goal.
 */
export function gridworld({x = 0, y = 0, cols, rows, cellSize = 48, walls = [], terminals = [], observed, values = [], valueColor = () => palette.white, trajectory = [], policy = [], agent: agentCell, showValues = true, valueFormat = v => Number(v.toFixed(2)), unknownFill = '#e7ebef', trajectoryMode = 'actual'} = {}) {
  if (!Number.isInteger(cols) || !Number.isInteger(rows) || cols < 1 || rows < 1) throw new RangeError('grid size must be positive integers');
  positive(cellSize, 'cell size');
  const key = ([c, r]) => `${c},${r}`;
  const valid = cell => Array.isArray(cell) && cell.length === 2 && cell.every(Number.isInteger) && cell[0] >= 0 && cell[0] < cols && cell[1] >= 0 && cell[1] < rows;
  for (const cells of [walls, terminals, observed ?? [], trajectory, agentCell ? [agentCell] : []]) for (const cell of cells) if (!valid(cell)) throw new RangeError('cell lies outside grid');
  const blocked = new Set(walls.map(key)), ends = new Set(terminals.map(key)), visible = observed === undefined ? null : new Set(observed.map(key));
  const seen = cell => !visible || visible.has(key(cell));
  const center = ([c, r]) => [x + (c + .5) * cellSize, y + (r + .5) * cellSize];
  let out = '';
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const cell = [c, r], px = x + c * cellSize, py = y + r * cellSize;
    if (!seen(cell)) {out += rect(px, py, cellSize, cellSize, {fill: unknownFill, radius: 0}); continue;}
    if (blocked.has(key(cell))) {out += rect(px, py, cellSize, cellSize, {fill: palette.ink, radius: 0}); continue;}
    const value = values[r]?.[c];
    if (value !== undefined && value !== null) finite(value, 'cell value');
    out += rect(px, py, cellSize, cellSize, {fill: value === undefined || value === null ? palette.white : valueColor(value), radius: 0});
    if (ends.has(key(cell))) out += rect(px + 5, py + 5, cellSize - 10, cellSize - 10, {fill: 'none', stroke: palette.ink, strokeWidth: 2, radius: 0});
    if (showValues && value !== undefined && value !== null) out += text(px + cellSize / 2, py + cellSize / 2 + 6, valueFormat(value), {size: 18, anchor: 'middle'});
  }
  for (let i = 1; i < trajectory.length; i++) {
    const from = trajectory[i - 1], to = trajectory[i];
    if (!seen(from) || !seen(to) || key(from) === key(to)) continue;
    out += arrow(...center(from), ...center(to), {mode: trajectoryMode, width: 3, head: Math.min(10, cellSize / 5)});
  }
  for (const {cell, direction} of policy) {
    if (!valid(cell)) throw new RangeError('policy cell lies outside grid');
    if (!seen(cell) || blocked.has(key(cell)) || ends.has(key(cell))) continue;
    if (!['up', 'down', 'left', 'right'].includes(direction)) throw new RangeError('policy direction must be up, down, left, or right');
    const [dx, dy] = {up:[0,-1],down:[0,1],left:[-1,0],right:[1,0]}[direction], [cx, cy] = center(cell), a = cellSize * .22;
    out += arrow(cx - dx * a, cy - dy * a, cx + dx * a, cy + dy * a, {color: palette.orange, head: 8});
  }
  if (agentCell && seen(agentCell)) out += agent(...center(agentCell), {radius: Math.min(12, cellSize * .24)});
  return `<g data-pattern="gridworld">${out}</g>`;
}

/** Each item is a supplied event, not an implied equal physical duration. */
export function timelineStrip({x = 28, y = 0, items, step = 72, draw, connect = true, mode = 'actual'} = {}) {
  if (!Array.isArray(items) || !items.length) throw new TypeError('timeline items are required');
  positive(step, 'timeline step');
  let out = '';
  items.forEach((item, i) => {
    const cx = x + step * i;
    if (connect && i) out += arrow(cx - step + 15, y, cx - 15, y, {mode});
    out += draw ? draw(item, cx, y, i) : circle(cx, y, 9, {fill: item.color ?? palette.blue});
    if (item.label !== undefined) out += text(cx, y + 34, item.label, {anchor: 'middle', size: 18});
  });
  return `<g data-pattern="timeline">${out}</g>`;
}
export function axes({x, y, width, height, xDomain, yDomain, xTicks = [], yTicks = [], xLabel, yLabel, tickFormat = v => String(v), grid = true} = {}) {
  positive(width, 'axes width'); positive(height, 'axes height');
  for (const [name, domain] of [['x', xDomain], ['y', yDomain]]) if (!Array.isArray(domain) || domain.length !== 2 || !domain.every(Number.isFinite) || domain[1] <= domain[0]) throw new RangeError(`${name} domain must be an increasing finite pair`);
  const X = value => x + (value - xDomain[0]) / (xDomain[1] - xDomain[0]) * width;
  const Y = value => y + height - (value - yDomain[0]) / (yDomain[1] - yDomain[0]) * height;
  let body = '';
  for (const value of xTicks) {
    if (value < xDomain[0] || value > xDomain[1]) throw new RangeError('x tick outside domain');
    if (grid) body += line(X(value), y, X(value), y + height, {width: 1});
    body += line(X(value), y + height, X(value), y + height + 5, {color: palette.ink});
    body += text(X(value), y + height + 26, tickFormat(value), {size: typography.tick, color: palette.muted, anchor: 'middle'});
  }
  for (const value of yTicks) {
    if (value < yDomain[0] || value > yDomain[1]) throw new RangeError('y tick outside domain');
    if (grid) body += line(x, Y(value), x + width, Y(value), {width: 1});
    body += line(x - 5, Y(value), x, Y(value), {color: palette.ink});
    body += text(x - 10, Y(value) + 5, tickFormat(value), {size: typography.tick, color: palette.muted, anchor: 'end'});
  }
  body += line(x, y, x, y + height, {color: palette.ink}) + line(x, y + height, x + width, y + height, {color: palette.ink});
  if (xLabel) body += text(x + width, y + height + 54, xLabel, {anchor: 'end', size: 18});
  if (yLabel) body += text(x, y - 18, yLabel, {size: 18});
  return {body, X, Y, xDomain: [...xDomain], yDomain: [...yDomain]};
}
/** No interpolation/smoothing. Missing values split the supplied path. */
export function curve(points, {X = v => v, Y = v => v, color = palette.blue, width = 3, dash = '', dots = false, radius = 4} = {}) {
  let d = '', active = false, marks = '';
  for (const point of points) {
    if (point === null || point === undefined) {active = false; continue;}
    if (!Array.isArray(point)) throw new TypeError('curve points must be finite [x,y] pairs or missing');
    if (point.some(v => v === null || v === undefined || Number.isNaN(v))) {active = false; continue;}
    if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)) throw new TypeError('curve points must be finite [x,y] pairs or missing');
    const [x, y] = [X(point[0]), Y(point[1])];
    d += `${active ? 'L' : 'M'}${number(x)} ${number(y)}`; active = true;
    if (dots) marks += circle(x, y, radius, {fill: color});
  }
  return `<path d="${d}" fill="none" stroke="${attr(color)}" stroke-width="${width}"${dashAttr(dash)}/>${marks}`;
}
export function legend({x = 28, y, items, step = 30, size = 18} = {}) {
  return items.map((item, i) => {
    const cy = y + step * i, color = item.color ?? (item.mode === 'imagined' ? palette.purple : palette.blue);
    const mark = item.type === 'dot' ? circle(x + 14, cy - 5, 6, {fill: color}) : item.type === 'arrow' ? arrow(x, cy - 5, x + 28, cy - 5, {color, mode: item.mode ?? 'actual', dash: item.dash}) : line(x, cy - 5, x + 28, cy - 5, {color, dash: item.dash, width: 3});
    return mark + text(x + 40, cy, item.label, {size});
  }).join('');
}

// Six compositions. They supply spatial organization, not a fixed paragraph grid.
export function taskView({world, annotation = '', ...spec}) {
  return frame({...spec, body: gridworld(world) + annotation});
}
export function temporalSnapshots({snapshots, x = 28, y = 100, step = 222, ...spec}) {
  const body = snapshots.map((snapshot, i) => `<g transform="translate(${number(x + i * step)} ${number(y)})">${text(0, 0, snapshot.label, {size: 20})}${snapshot.body}</g>`).join('');
  return frame({...spec, body});
}
export function valuePolicyMap({world, key = '', ...spec}) {
  return frame({...spec, body: gridworld(world) + key});
}
export function updateDataflow({nodes, edges, ...spec}) {
  const byId = new Map(nodes.map(node => [node.id, node]));
  if (byId.size !== nodes.length) throw new TypeError('dataflow node IDs must be unique');
  const body = edges.map(edge => {
    const source = byId.get(edge.from), target = byId.get(edge.to);
    if (!source || !target) throw new TypeError('dataflow edge references unknown node');
    return arrow(edge.x1 ?? source.x, edge.y1 ?? source.y, edge.x2 ?? target.x, edge.y2 ?? target.y, edge);
  }).join('') + nodes.map(node => node.body ?? agent(node.x, node.y, {color: node.color, label: node.label})).join('');
  return frame({...spec, body});
}
export function comparisonGeometry({panels, ...spec}) {
  return frame({...spec, body: panels.map(item => `<g transform="translate(${number(item.x ?? 0)} ${number(item.y ?? 0)})">${item.body}</g>`).join('')});
}
export function experimentEvidence({plot, series, annotation = '', ...spec}) {
  if (!['exact', 'measured', 'schematic'].includes(spec.kind)) throw new TypeError('experimentEvidence requires an explicit evidence kind');
  const coordinate = axes(plot);
  return frame({...spec, body: coordinate.body + series.map(series => curve(series.points, {...coordinate, ...series})).join('') + annotation});
}
