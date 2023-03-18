/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var geometryGradientBar = /* color: #d63000 */ee.Geometry.LineString(
        [[-104.93453530069758, 39.7312002711519],
         [-104.90861424118077, 39.732718726425524]]),
    geometryScaleBar = /* color: #98ff00 */ee.Geometry.LineString(
        [[-104.93333776664349, 39.72560688829384],
         [-104.90853336034098, 39.72527685940222]]);
/***** End of imports. If edited, may not auto-convert in the playground. *****/
// File: style-test-gradientbar
// Package sources: https://code.earthengine.google.com/?accept_repo=users/gena/packages

var palettes = require('users/gena/packages:palettes');
var style = require('users/gena/packages:style');
var utils = require('users/gena/packages:utils');

var dem = ee.Image("USGS/NED");

var minMax = dem.reduceRegion(ee.Reducer.percentile([1, 99]), Map.getBounds(true), Map.getScale()).values();

var min = ee.Number(minMax.get(0));
var max = ee.Number(minMax.get(1));

var exaggregate = 5;
var weight = 1.5;
var azimuth = 0;
var zenith = 45;

// Paired
var palette = palettes.colorbrewer.Paired[12];
var styled = dem.visualize({ min: min, max: max, palette: palette });
Map.addLayer(styled, {}, 'Paired, styled', false);

var hillshaded = utils.hillshadeit(styled, dem, weight, exaggregate, azimuth, zenith);
Map.addLayer(hillshaded, {}, 'Paired, hillshaded');

var textProperties = { fontSize:16, textColor: '000000', outlineColor: 'ffffff', outlineWidth: 2, outlineOpacity: 0.6 };

// dim background
Map.addLayer(ee.Image(1), {palette: ['ffffff']}, 'white', true, 0.5);

// add a gradient bar
var labels = ee.List.sequence(min, max, max.subtract(min).divide(5));
var gradient = style.GradientBar.draw(geometryGradientBar, {
  min: min, max: max, palette: palette, labels: labels, format: '%.0f', text: textProperties
});
Map.addLayer(gradient, {}, 'gradient bar (DEM)');

// add a scale bar
var scale = style.ScaleBar.draw(geometryScaleBar, {
  steps:4, palette: ['5ab4ac', 'f5f5f5'], multiplier: 1000, format: '%.1f', units: 'km', text: textProperties
});
Map.addLayer(scale, {}, 'scale bar');

