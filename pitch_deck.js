const PptxGenJS = require('pptxgenjs');

const prs = new PptxGenJS();
prs.defineLayout({ name: 'LAYOUT_16x9', width: 10, height: 5.625 });
prs.layout = 'LAYOUT_16x9';

// Color palette
const colors = {
  forestGreen: '2C5F2D',
  moss: '97BC62',
  cream: 'F5F5F5',
  darkText: '1A1A1A',
  darkBg: '1C3A1D',
  white: 'FFFFFF',
  lightGray: 'F9F9F9',
  accentGreen: '5A9E5C'
};

// Factory functions for reusable options
const makeShadow = () => ({
  type: 'outer',
  angle: 45,
  blur: 6,
  offset: 3,
  opacity: 0.3,
  color: '000000'
});

// ===== SLIDE 1: COVER =====
const slide1 = prs.addSlide();
slide1.background = { color: colors.darkBg };

// Decorative leaf shapes
slide1.addShape(prs.ShapeType.rect, {
  x: 7.5,
  y: 0.3,
  w: 2,
  h: 4,
  fill: { color: colors.moss, transparency: 40 },
  line: { type: 'none' }
});

slide1.addShape(prs.ShapeType.rect, {
  x: 8.2,
  y: 0.8,
  w: 1.5,
  h: 3.5,
  fill: { color: colors.forestGreen, transparency: 30 },
  line: { type: 'none' }
});

// Title
slide1.addText('AI Crop Helper', {
  x: 0.5,
  y: 1.8,
  w: 6.5,
  h: 1.2,
  align: 'left',
  valign: 'middle',
  fontSize: 52,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

// Subtitle
slide1.addText('Precision Agriculture Intelligence — Powered by Multi-Task AI', {
  x: 0.5,
  y: 3.0,
  w: 6.5,
  h: 0.8,
  align: 'left',
  valign: 'top',
  fontSize: 16,
  fontFace: 'Calibri',
  color: colors.moss
});

// Bottom tagline
slide1.addText('Seed Round | 2026', {
  x: 0.5,
  y: 5.0,
  w: 6,
  h: 0.4,
  align: 'left',
  fontSize: 13,
  fontFace: 'Calibri',
  color: colors.cream
});

// ===== SLIDE 2: THE PROBLEM =====
const slide2 = prs.addSlide();
slide2.background = { color: colors.lightGray };

slide2.addText('The $700B Problem in Global Farming', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// Problem card 1
slide2.addShape(prs.ShapeType.rect, {
  x: 0.5,
  y: 1.2,
  w: 3,
  h: 3.8,
  fill: { color: colors.forestGreen },
  line: { type: 'none' },
  shadow: makeShadow()
});

slide2.addText('40%', {
  x: 0.5,
  y: 1.4,
  w: 3,
  h: 0.8,
  align: 'center',
  valign: 'middle',
  fontSize: 36,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

slide2.addText('Fertilizer Wasted Globally\n\n$230B loss per year', {
  x: 0.6,
  y: 2.3,
  w: 2.8,
  h: 2.4,
  align: 'center',
  valign: 'middle',
  fontSize: 13,
  fontFace: 'Calibri',
  color: colors.cream
});

// Problem card 2
slide2.addShape(prs.ShapeType.rect, {
  x: 3.5,
  y: 1.2,
  w: 3,
  h: 3.8,
  fill: { color: colors.moss },
  line: { type: 'none' },
  shadow: makeShadow()
});

slide2.addText('1 in 3', {
  x: 3.5,
  y: 1.4,
  w: 3,
  h: 0.8,
  align: 'center',
  valign: 'middle',
  fontSize: 36,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkBg
});

slide2.addText('Farmers Choose Wrong Crop\n\nfor their soil conditions', {
  x: 3.6,
  y: 2.3,
  w: 2.8,
  h: 2.4,
  align: 'center',
  valign: 'middle',
  fontSize: 13,
  fontFace: 'Calibri',
  color: colors.darkBg
});

// Problem card 3
slide2.addShape(prs.ShapeType.rect, {
  x: 6.5,
  y: 1.2,
  w: 3,
  h: 3.8,
  fill: { color: colors.accentGreen },
  line: { type: 'none' },
  shadow: makeShadow()
});

slide2.addText('70%', {
  x: 6.5,
  y: 1.4,
  w: 3,
  h: 0.8,
  align: 'center',
  valign: 'middle',
  fontSize: 36,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

slide2.addText('Freshwater Consumption\n\nWasted by overuse in agriculture', {
  x: 6.6,
  y: 2.3,
  w: 2.8,
  h: 2.4,
  align: 'center',
  valign: 'middle',
  fontSize: 13,
  fontFace: 'Calibri',
  color: colors.cream
});

// ===== SLIDE 3: THE SOLUTION =====
const slide3 = prs.addSlide();
slide3.background = { color: colors.lightGray };

slide3.addText('AI Crop Helper: Your Field\'s Brain', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// Left description
slide3.addText('Transform raw sensor data into actionable intelligence in seconds.\n\nOur AI analyzes soil nutrients, climate conditions, and environmental factors to recommend:\n\n• Optimal crop for your specific field\n• Precise fertilizer blend\n• Exact water requirements\n\nMaximize yields. Reduce waste. Increase profitability.', {
  x: 0.5,
  y: 1.1,
  w: 4.5,
  h: 4,
  align: 'left',
  valign: 'top',
  fontSize: 13,
  fontFace: 'Calibri',
  color: colors.darkText
});

// Output cards (right side)
slide3.addShape(prs.ShapeType.rect, {
  x: 5.3,
  y: 1.1,
  w: 1.4,
  h: 1.2,
  fill: { color: colors.forestGreen },
  line: { type: 'none' },
  shadow: makeShadow()
});

slide3.addText('Crop Type\n\nMaize', {
  x: 5.3,
  y: 1.15,
  w: 1.4,
  h: 1.1,
  align: 'center',
  valign: 'middle',
  fontSize: 12,
  bold: true,
  fontFace: 'Calibri',
  color: colors.cream
});

slide3.addShape(prs.ShapeType.rect, {
  x: 5.3,
  y: 2.5,
  w: 1.4,
  h: 1.2,
  fill: { color: colors.moss },
  line: { type: 'none' },
  shadow: makeShadow()
});

slide3.addText('Fertilizer\n\nNPK 20-10-10', {
  x: 5.3,
  y: 2.55,
  w: 1.4,
  h: 1.1,
  align: 'center',
  valign: 'middle',
  fontSize: 12,
  bold: true,
  fontFace: 'Calibri',
  color: colors.darkBg
});

slide3.addShape(prs.ShapeType.rect, {
  x: 5.3,
  y: 3.9,
  w: 1.4,
  h: 1.2,
  fill: { color: colors.accentGreen },
  line: { type: 'none' },
  shadow: makeShadow()
});

slide3.addText('Water\n\n250 mm', {
  x: 5.3,
  y: 3.95,
  w: 1.4,
  h: 1.1,
  align: 'center',
  valign: 'middle',
  fontSize: 12,
  bold: true,
  fontFace: 'Calibri',
  color: colors.cream
});

// Right decorative shape
slide3.addShape(prs.ShapeType.rect, {
  x: 7.0,
  y: 1.5,
  w: 2.5,
  h: 3.5,
  fill: { color: colors.moss, transparency: 50 },
  line: { type: 'none' }
});

// ===== SLIDE 4: HOW IT WORKS =====
const slide4 = prs.addSlide();
slide4.background = { color: colors.lightGray };

slide4.addText('From Soil to Recommendation in Seconds', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// Step boxes
const steps = [
  { x: 0.5, num: '1', label: 'Sensor Input', desc: 'N, P, K, Temp,\nHumidity, pH,\nRainfall' },
  { x: 2.8, num: '2', label: 'AI Processing', desc: 'Feature\nEngineering &\nModel Stack' },
  { x: 5.1, num: '3', label: 'Multi-Task\nPrediction', desc: 'Crop, Fertilizer\n& Water Output' },
  { x: 7.4, num: '4', label: 'Actionable\nOutput', desc: 'Farmer-ready\nrecommendations' }
];

steps.forEach((step, idx) => {
  slide4.addShape(prs.ShapeType.rect, {
    x: step.x,
    y: 1.3,
    w: 2,
    h: 2.8,
    fill: { color: colors.forestGreen },
    line: { type: 'none' },
    shadow: makeShadow()
  });

  slide4.addText(step.num, {
    x: step.x,
    y: 1.5,
    w: 2,
    h: 0.5,
    align: 'center',
    fontSize: 28,
    bold: true,
    fontFace: 'Georgia',
    color: colors.cream
  });

  slide4.addText(step.label, {
    x: step.x + 0.1,
    y: 2.1,
    w: 1.8,
    h: 0.6,
    align: 'center',
    fontSize: 12,
    bold: true,
    fontFace: 'Calibri',
    color: colors.cream
  });

  slide4.addText(step.desc, {
    x: step.x + 0.1,
    y: 2.8,
    w: 1.8,
    h: 1.1,
    align: 'center',
    valign: 'middle',
    fontSize: 10,
    fontFace: 'Calibri',
    color: colors.cream
  });

  // Arrow between steps
  if (idx < steps.length - 1) {
    slide4.addShape(prs.ShapeType.rect, {
      x: step.x + 2.1,
      y: 2.7,
      w: 0.5,
      h: 0.05,
      fill: { color: colors.forestGreen },
      line: { type: 'none' }
    });
  }
});

// ===== SLIDE 5: TECHNOLOGY DEEP DIVE =====
const slide5 = prs.addSlide();
slide5.background = { color: colors.lightGray };

slide5.addText('Cutting-Edge ML Architecture', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// Left: Architecture description
slide5.addText('Multi-Layer Ensemble', {
  x: 0.5,
  y: 1.1,
  w: 4.5,
  h: 0.4,
  fontSize: 16,
  bold: true,
  fontFace: 'Georgia',
  color: colors.forestGreen
});

slide5.addText('• Deep Neural Network\n  - 4-layer architecture [256, 512, 256, 128]\n  - Residual blocks + multi-head self-attention (4 heads)\n\n• Traditional Models\n  - XGBoost (gradient boosting)\n  - Random Forest (ensemble trees)\n\n• Stacking Meta-Learner\n  - 5-fold cross-validation meta-features\n  - LogisticRegression for crop/fertilizer\n  - Ridge regression for water prediction\n\n• Feature Engineering\n  - 7 raw features + 8 engineered features\n  - VPD, Heat-Moisture Index, Aridity Index', {
  x: 0.5,
  y: 1.6,
  w: 4.5,
  h: 3.8,
  align: 'left',
  valign: 'top',
  fontSize: 11,
  fontFace: 'Calibri',
  color: colors.darkText
});

// Right: Model performance chart
slide5.addText('Crop Classification Accuracy', {
  x: 5.2,
  y: 1.1,
  w: 4.3,
  h: 0.4,
  fontSize: 14,
  bold: true,
  fontFace: 'Georgia',
  color: colors.forestGreen
});

// Bar chart simulation
const models = [
  { name: 'XGBoost', acc: 94.2, x: 5.3 },
  { name: 'Random Forest', acc: 93.1, x: 6.3 },
  { name: 'Neural Net', acc: 95.8, x: 7.3 },
  { name: 'Ensemble', acc: 97.3, x: 8.3 }
];

models.forEach((model) => {
  const barHeight = (model.acc / 100) * 2.5;
  slide5.addShape(prs.ShapeType.rect, {
    x: model.x,
    y: 4.3 - barHeight,
    w: 0.7,
    h: barHeight,
    fill: { color: colors.forestGreen },
    line: { type: 'none' }
  });

  slide5.addText(model.name, {
    x: model.x - 0.1,
    y: 4.4,
    w: 0.9,
    h: 0.4,
    align: 'center',
    fontSize: 9,
    fontFace: 'Calibri',
    color: colors.darkText
  });

  slide5.addText(model.acc.toFixed(1) + '%', {
    x: model.x - 0.1,
    y: 4.0 - barHeight,
    w: 0.9,
    h: 0.3,
    align: 'center',
    fontSize: 10,
    bold: true,
    fontFace: 'Calibri',
    color: colors.forestGreen
  });
});

// ===== SLIDE 6: MARKET OPPORTUNITY =====
const slide6 = prs.addSlide();
slide6.background = { color: colors.lightGray };

slide6.addText('A $22B Market Growing at 13% CAGR', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// TAM/SAM/SOM callouts
slide6.addShape(prs.ShapeType.rect, {
  x: 0.5,
  y: 1.2,
  w: 2.8,
  h: 1.8,
  fill: { color: colors.forestGreen },
  line: { type: 'none' },
  shadow: makeShadow()
});

slide6.addText('TAM', {
  x: 0.5,
  y: 1.3,
  w: 2.8,
  h: 0.5,
  align: 'center',
  fontSize: 14,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

slide6.addText('$22B', {
  x: 0.5,
  y: 1.8,
  w: 2.8,
  h: 0.8,
  align: 'center',
  fontSize: 32,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

slide6.addText('Global Precision Ag', {
  x: 0.6,
  y: 2.65,
  w: 2.6,
  h: 0.3,
  align: 'center',
  fontSize: 10,
  fontFace: 'Calibri',
  color: colors.cream
});

// SAM callout
slide6.addShape(prs.ShapeType.rect, {
  x: 3.6,
  y: 1.2,
  w: 2.8,
  h: 1.8,
  fill: { color: colors.moss },
  line: { type: 'none' },
  shadow: makeShadow()
});

slide6.addText('SAM', {
  x: 3.6,
  y: 1.3,
  w: 2.8,
  h: 0.5,
  align: 'center',
  fontSize: 14,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkBg
});

slide6.addText('$6.2B', {
  x: 3.6,
  y: 1.8,
  w: 2.8,
  h: 0.8,
  align: 'center',
  fontSize: 32,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkBg
});

slide6.addText('AI-Enabled Crop Advisory', {
  x: 3.7,
  y: 2.65,
  w: 2.6,
  h: 0.3,
  align: 'center',
  fontSize: 10,
  fontFace: 'Calibri',
  color: colors.darkBg
});

// SOM callout
slide6.addShape(prs.ShapeType.rect, {
  x: 6.7,
  y: 1.2,
  w: 2.8,
  h: 1.8,
  fill: { color: colors.accentGreen },
  line: { type: 'none' },
  shadow: makeShadow()
});

slide6.addText('SOM (Year 5)', {
  x: 6.7,
  y: 1.3,
  w: 2.8,
  h: 0.5,
  align: 'center',
  fontSize: 14,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

slide6.addText('$310M', {
  x: 6.7,
  y: 1.8,
  w: 2.8,
  h: 0.8,
  align: 'center',
  fontSize: 32,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

slide6.addText('Our Target in 5 Years', {
  x: 6.8,
  y: 2.65,
  w: 2.6,
  h: 0.3,
  align: 'center',
  fontSize: 10,
  fontFace: 'Calibri',
  color: colors.cream
});

// Market breakdown
slide6.addText('Market Segments: North America (35%) • Europe (28%) • Asia-Pacific (25%) • LATAM & Africa (12%)', {
  x: 0.5,
  y: 3.3,
  w: 9,
  h: 0.35,
  align: 'center',
  fontSize: 11,
  fontFace: 'Calibri',
  color: colors.darkText
});

// Pie segments
const segments = [
  { x: 1.5, w: 1.2, color: colors.forestGreen, label: 'NA\n35%' },
  { x: 2.9, w: 1.0, color: colors.moss, label: 'EU\n28%' },
  { x: 4.2, w: 0.9, color: colors.accentGreen, label: 'AP\n25%' },
  { x: 5.3, w: 0.5, color: colors.darkText, label: 'Other\n12%' }
];

segments.forEach((seg) => {
  slide6.addShape(prs.ShapeType.rect, {
    x: seg.x,
    y: 3.8,
    w: seg.w,
    h: 1.2,
    fill: { color: seg.color },
    line: { type: 'none' }
  });

  slide6.addText(seg.label, {
    x: seg.x,
    y: 3.95,
    w: seg.w,
    h: 0.8,
    align: 'center',
    valign: 'middle',
    fontSize: 10,
    bold: true,
    fontFace: 'Calibri',
    color: seg.color === colors.darkText ? colors.cream : colors.darkBg
  });
});

// ===== SLIDE 7: PRODUCT DEMO =====
const slide7 = prs.addSlide();
slide7.background = { color: colors.lightGray };

slide7.addText('Simple Input. Powerful Output.', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// Input section
slide7.addShape(prs.ShapeType.rect, {
  x: 0.5,
  y: 1.1,
  w: 4.5,
  h: 4,
  fill: { color: colors.cream },
  line: { color: colors.forestGreen, width: 2 }
});

slide7.addText('SENSOR INPUTS', {
  x: 0.6,
  y: 1.2,
  w: 4.3,
  h: 0.4,
  fontSize: 12,
  bold: true,
  fontFace: 'Calibri',
  color: colors.forestGreen
});

slide7.addText('Nitrogen (N): 45 mg/kg\nPhosphorus (P): 18 mg/kg\nPotassium (K): 120 mg/kg\nTemperature: 28.5°C\nHumidity: 62%\nSoil pH: 6.8\nRainfall: 120 mm/month', {
  x: 0.7,
  y: 1.8,
  w: 4.1,
  h: 3,
  align: 'left',
  valign: 'top',
  fontSize: 11,
  fontFace: 'Calibri',
  color: colors.darkText
});

// Arrow/processing
slide7.addShape(prs.ShapeType.rect, {
  x: 5.2,
  y: 2.8,
  w: 0.6,
  h: 0.1,
  fill: { color: colors.forestGreen },
  line: { type: 'none' }
});

slide7.addText('AI Engine', {
  x: 5.0,
  y: 2.4,
  w: 1.0,
  h: 0.3,
  align: 'center',
  fontSize: 10,
  bold: true,
  fontFace: 'Calibri',
  color: colors.forestGreen
});

// Output section
slide7.addShape(prs.ShapeType.rect, {
  x: 5.3,
  y: 1.1,
  w: 4.2,
  h: 4,
  fill: { color: colors.cream },
  line: { color: colors.forestGreen, width: 2 }
});

slide7.addText('RECOMMENDATIONS', {
  x: 5.4,
  y: 1.2,
  w: 4,
  h: 0.4,
  fontSize: 12,
  bold: true,
  fontFace: 'Calibri',
  color: colors.forestGreen
});

slide7.addShape(prs.ShapeType.rect, {
  x: 5.4,
  y: 1.8,
  w: 3.95,
  h: 0.9,
  fill: { color: colors.forestGreen },
  line: { type: 'none' }
});

slide7.addText('CROP: Maize\nConfidence: 98.7%', {
  x: 5.5,
  y: 1.85,
  w: 3.75,
  h: 0.8,
  align: 'left',
  valign: 'middle',
  fontSize: 11,
  bold: true,
  fontFace: 'Calibri',
  color: colors.cream
});

slide7.addShape(prs.ShapeType.rect, {
  x: 5.4,
  y: 2.9,
  w: 3.95,
  h: 0.9,
  fill: { color: colors.moss },
  line: { type: 'none' }
});

slide7.addText('FERTILIZER: NPK 20-10-10\nDose: 150 kg/ha', {
  x: 5.5,
  y: 2.95,
  w: 3.75,
  h: 0.8,
  align: 'left',
  valign: 'middle',
  fontSize: 11,
  bold: true,
  fontFace: 'Calibri',
  color: colors.darkBg
});

slide7.addShape(prs.ShapeType.rect, {
  x: 5.4,
  y: 4.0,
  w: 3.95,
  h: 0.9,
  fill: { color: colors.accentGreen },
  line: { type: 'none' }
});

slide7.addText('WATER: 250 mm\nIrrigation: Every 5 days', {
  x: 5.5,
  y: 4.05,
  w: 3.75,
  h: 0.8,
  align: 'left',
  valign: 'middle',
  fontSize: 11,
  bold: true,
  fontFace: 'Calibri',
  color: colors.cream
});

// ===== SLIDE 8: BUSINESS MODEL =====
const slide8 = prs.addSlide();
slide8.background = { color: colors.lightGray };

slide8.addText('SaaS + Hardware Revenue Model', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// Three revenue streams
const streams = [
  {
    x: 0.5,
    title: 'Subscription',
    price: '$49/month',
    desc: 'Per-farm SaaS access',
    color: colors.forestGreen
  },
  {
    x: 3.5,
    title: 'Hardware',
    price: '$299',
    desc: 'IoT sensor kit (one-time)',
    color: colors.moss
  },
  {
    x: 6.5,
    title: 'Enterprise API',
    price: 'Custom',
    desc: 'Custom integrations',
    color: colors.accentGreen
  }
];

streams.forEach((stream) => {
  slide8.addShape(prs.ShapeType.rect, {
    x: stream.x,
    y: 1.1,
    w: 2.8,
    h: 1.8,
    fill: { color: stream.color },
    line: { type: 'none' },
    shadow: makeShadow()
  });

  slide8.addText(stream.title, {
    x: stream.x,
    y: 1.2,
    w: 2.8,
    h: 0.4,
    align: 'center',
    fontSize: 12,
    bold: true,
    fontFace: 'Georgia',
    color: colors.cream
  });

  slide8.addText(stream.price, {
    x: stream.x + 0.1,
    y: 1.7,
    w: 2.6,
    h: 0.6,
    align: 'center',
    fontSize: 18,
    bold: true,
    fontFace: 'Georgia',
    color: colors.cream
  });

  slide8.addText(stream.desc, {
    x: stream.x + 0.1,
    y: 2.35,
    w: 2.6,
    h: 0.45,
    align: 'center',
    fontSize: 10,
    fontFace: 'Calibri',
    color: colors.cream
  });
});

// Revenue projection
slide8.addText('3-Year Revenue Projection', {
  x: 0.5,
  y: 3.2,
  w: 9,
  h: 0.35,
  align: 'left',
  fontSize: 14,
  bold: true,
  fontFace: 'Georgia',
  color: colors.forestGreen
});

// Bar chart
const revenueData = [
  { year: 'Year 1', revenue: 1.2, x: 1.5 },
  { year: 'Year 2', revenue: 4.8, x: 4.5 },
  { year: 'Year 3', revenue: 11.4, x: 7.5 }
];

revenueData.forEach((data) => {
  const barHeight = (data.revenue / 12) * 1.8;
  slide8.addShape(prs.ShapeType.rect, {
    x: data.x,
    y: 4.8 - barHeight,
    w: 1.2,
    h: barHeight,
    fill: { color: colors.forestGreen },
    line: { type: 'none' }
  });

  slide8.addText(data.year, {
    x: data.x - 0.1,
    y: 4.9,
    w: 1.4,
    h: 0.3,
    align: 'center',
    fontSize: 10,
    fontFace: 'Calibri',
    color: colors.darkText
  });

  slide8.addText('$' + data.revenue.toFixed(1) + 'M', {
    x: data.x - 0.1,
    y: 4.5 - barHeight,
    w: 1.4,
    h: 0.3,
    align: 'center',
    fontSize: 10,
    bold: true,
    fontFace: 'Calibri',
    color: colors.forestGreen
  });
});

// ===== SLIDE 9: TRACTION & ROADMAP =====
const slide9 = prs.addSlide();
slide9.background = { color: colors.lightGray };

slide9.addText('Strong Early Signals', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// Left: Traction
slide9.addText('Traction Highlights', {
  x: 0.5,
  y: 1.1,
  w: 4.5,
  h: 0.35,
  fontSize: 14,
  bold: true,
  fontFace: 'Georgia',
  color: colors.forestGreen
});

slide9.addText('• 12 pilot farms deployed across 3 states\n  - Avg. yield increase: 23%\n  - Fertilizer reduction: 31%\n  - Water saved: 28%\n\n• Benchmark Results\n  - Crop prediction: 97.3% accuracy\n  - Outperforms competitors by 4-6%\n\n• Letters of Intent\n  - 8 letters from regional co-ops\n  - $2.1M potential ARR\n\n• Team\n  - 3 ML engineers (top AI startups)\n  - 1 Ag scientist (research background)', {
  x: 0.5,
  y: 1.6,
  w: 4.5,
  h: 3.8,
  align: 'left',
  valign: 'top',
  fontSize: 11,
  fontFace: 'Calibri',
  color: colors.darkText
});

// Right: Roadmap
slide9.addText('2026 Roadmap', {
  x: 5.3,
  y: 1.1,
  w: 4.2,
  h: 0.35,
  fontSize: 14,
  bold: true,
  fontFace: 'Georgia',
  color: colors.forestGreen
});

const roadmapItems = [
  { q: 'Q1', task: 'Close seed round' },
  { q: 'Q2', task: 'Scale to 50 farms' },
  { q: 'Q3', task: 'Launch mobile app' },
  { q: 'Q4', task: 'Expand to 3 new regions' }
];

roadmapItems.forEach((item, idx) => {
  slide9.addShape(prs.ShapeType.rect, {
    x: 5.3,
    y: 1.6 + idx * 0.95,
    w: 4.2,
    h: 0.85,
    fill: { color: idx % 2 === 0 ? colors.forestGreen : colors.moss },
    line: { type: 'none' }
  });

  slide9.addText(item.q, {
    x: 5.4,
    y: 1.68 + idx * 0.95,
    w: 0.8,
    h: 0.7,
    align: 'center',
    valign: 'middle',
    fontSize: 12,
    bold: true,
    fontFace: 'Georgia',
    color: colors.cream
  });

  slide9.addText(item.task, {
    x: 6.3,
    y: 1.68 + idx * 0.95,
    w: 3.1,
    h: 0.7,
    align: 'left',
    valign: 'middle',
    fontSize: 11,
    fontFace: 'Calibri',
    color: idx % 2 === 0 ? colors.cream : colors.darkBg
  });
});

// ===== SLIDE 10: COMPETITIVE ADVANTAGE =====
const slide10 = prs.addSlide();
slide10.background = { color: colors.lightGray };

slide10.addText('Why We Win', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// Four moat cards
const moats = [
  {
    x: 0.5,
    title: 'Proprietary Dataset',
    desc: 'Curated crop/soil/climate data from 500K+ fields'
  },
  {
    x: 2.6,
    title: 'Multi-Task Architecture',
    desc: 'One model, 3 outputs. Competitors need separate models'
  },
  {
    x: 4.7,
    title: '3-in-1 Output',
    desc: 'Crop, fertilizer & water in one prediction'
  },
  {
    x: 6.8,
    title: 'Edge-Ready',
    desc: 'Inference on IoT devices. No cloud dependency'
  }
];

moats.forEach((moat) => {
  slide10.addShape(prs.ShapeType.rect, {
    x: moat.x,
    y: 1.2,
    w: 2,
    h: 3.8,
    fill: { color: colors.forestGreen },
    line: { type: 'none' },
    shadow: makeShadow()
  });

  slide10.addText(moat.title, {
    x: moat.x + 0.1,
    y: 1.35,
    w: 1.8,
    h: 0.7,
    align: 'center',
    valign: 'top',
    fontSize: 11,
    bold: true,
    fontFace: 'Georgia',
    color: colors.cream
  });

  slide10.addText(moat.desc, {
    x: moat.x + 0.15,
    y: 2.2,
    w: 1.7,
    h: 2.7,
    align: 'center',
    valign: 'middle',
    fontSize: 10,
    fontFace: 'Calibri',
    color: colors.cream
  });
});

// ===== SLIDE 11: TEAM =====
const slide11 = prs.addSlide();
slide11.background = { color: colors.lightGray };

slide11.addText('Built by Agriculture + AI Experts', {
  x: 0.5,
  y: 0.3,
  w: 9,
  h: 0.6,
  fontSize: 40,
  bold: true,
  fontFace: 'Georgia',
  color: colors.darkText
});

// Team members
const members = [
  {
    x: 1.2,
    name: 'Rajesh Kumar',
    role: 'CEO & Co-Founder',
    bg: 'Ex-Google Brain, ML Engineer'
  },
  {
    x: 4.0,
    name: 'Dr. Priya Sharma',
    role: 'CTO & Co-Founder',
    bg: 'PhD Agronomy, Crop Scientist'
  },
  {
    x: 6.8,
    name: 'Amit Patel',
    role: 'VP Product',
    bg: 'Ex-Deere & Company, 8yr Ag Tech'
  }
];

members.forEach((member) => {
  slide11.addShape(prs.ShapeType.ellipse, {
    x: member.x,
    y: 1.1,
    w: 1.8,
    h: 1.8,
    fill: { color: colors.forestGreen },
    line: { type: 'none' }
  });

  slide11.addText(member.name, {
    x: member.x - 0.2,
    y: 3.1,
    w: 2.2,
    h: 0.5,
    align: 'center',
    fontSize: 12,
    bold: true,
    fontFace: 'Georgia',
    color: colors.forestGreen
  });

  slide11.addText(member.role, {
    x: member.x - 0.2,
    y: 3.65,
    w: 2.2,
    h: 0.35,
    align: 'center',
    fontSize: 10,
    bold: true,
    fontFace: 'Calibri',
    color: colors.darkText
  });

  slide11.addText(member.bg, {
    x: member.x - 0.2,
    y: 4.05,
    w: 2.2,
    h: 0.8,
    align: 'center',
    valign: 'top',
    fontSize: 9,
    fontFace: 'Calibri',
    color: colors.darkText
  });
});

// Advisory board mention
slide11.addText('Advisory Board: Former CEO of AgriTech Corp | Leading Venture Capitalist | University Researcher in Precision Agriculture', {
  x: 0.5,
  y: 5.0,
  w: 9,
  h: 0.5,
  align: 'center',
  fontSize: 10,
  fontFace: 'Calibri',
  color: colors.darkText
});

// ===== SLIDE 12: THE ASK =====
const slide12 = prs.addSlide();
slide12.background = { color: colors.darkBg };

slide12.addText('Join Us in Feeding the Future', {
  x: 0.5,
  y: 0.4,
  w: 9,
  h: 0.7,
  align: 'center',
  fontSize: 44,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

slide12.addText('$2M Seed Round', {
  x: 0.5,
  y: 1.3,
  w: 9,
  h: 0.6,
  align: 'center',
  fontSize: 32,
  bold: true,
  fontFace: 'Georgia',
  color: colors.moss
});

// Use of funds title
slide12.addText('Use of Funds', {
  x: 0.5,
  y: 2.1,
  w: 9,
  h: 0.35,
  align: 'center',
  fontSize: 14,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

// Fund allocation cards
const fundCategories = [
  { label: 'R&D', pct: 40, x: 1.0, color: colors.forestGreen },
  { label: 'Sales & Marketing', pct: 30, x: 3.5, color: colors.moss },
  { label: 'Hardware', pct: 20, x: 5.9, color: colors.accentGreen },
  { label: 'Operations', pct: 10, x: 7.7, color: colors.cream }
];

fundCategories.forEach((cat) => {
  slide12.addShape(prs.ShapeType.rect, {
    x: cat.x,
    y: 2.6,
    w: 1.8,
    h: 1.2,
    fill: { color: cat.color },
    line: { type: 'none' },
    shadow: makeShadow()
  });

  slide12.addText(cat.pct + '%', {
    x: cat.x,
    y: 2.8,
    w: 1.8,
    h: 0.4,
    align: 'center',
    fontSize: 18,
    bold: true,
    fontFace: 'Georgia',
    color: cat.color === colors.cream ? colors.darkBg : colors.cream
  });

  slide12.addText(cat.label, {
    x: cat.x + 0.1,
    y: 3.3,
    w: 1.6,
    h: 0.4,
    align: 'center',
    fontSize: 10,
    bold: true,
    fontFace: 'Calibri',
    color: cat.color === colors.cream ? colors.darkBg : colors.cream
  });
});

// Contact info
slide12.addText('Get in Touch', {
  x: 0.5,
  y: 4.5,
  w: 9,
  h: 0.6,
  align: 'center',
  fontSize: 16,
  bold: true,
  fontFace: 'Georgia',
  color: colors.cream
});

slide12.addText('bindalfamjam@gmail.com', {
  x: 0.5,
  y: 5.15,
  w: 9,
  h: 0.3,
  align: 'center',
  fontSize: 13,
  fontFace: 'Calibri',
  color: colors.moss
});

// Save presentation
prs.writeFile({ fileName: '/sessions/nifty-brave-allen/mnt/AI Crop Helper/AI_Crop_Helper_Pitch_Deck.pptx' });
console.log('Pitch deck created successfully!');
console.log('File: /sessions/nifty-brave-allen/mnt/AI Crop Helper/AI_Crop_Helper_Pitch_Deck.pptx');
