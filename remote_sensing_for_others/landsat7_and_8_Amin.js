// Landsat 8
/**
*
* This is a copy of L7_C2L2_Scaled adaopted for L8.
* https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands
*/

// var SF = ee.FeatureCollection(intersect_Grant_2Cols);

// var SF = ee.FeatureCollection(intersect_Grant_Irr_2008_2018_2cols_10);
// var outfile_name_prefix = 'L5_T1C2L2_Scaled_int_Grant_Irr_2008_2018_2cols_10';
// var wstart_date = '2008-01-01';
// var wend_date = '2022-01-01';


// var SF = ee.FeatureCollection(Walla2015);
// var outfile_name_prefix = 'L8_T1C2L2_Scaled_Walla2015_';
// var wstart_date = '2014-01-01';
// var wend_date = '2017-01-01';

var SF = ee.FeatureCollection(AdamBenton2016);
var outfile_name_prefix = 'L8_T1C2L2_Scaled_AdamBenton2016_';
var wstart_date = '2015-01-01';
var wend_date = '2017-01-01';

// var SF = ee.FeatureCollection(Grant2017);
// var outfile_name_prefix = 'L8_T1C2L2_Scaled_Grant2017_';
// var wstart_date = '2016-01-01';
// var wend_date = '2018-01-01';

// var SF = ee.FeatureCollection(FranklinYakima2018);
// var outfile_name_prefix = 'L8_T1C2L2_Scaled_FranklinYakima2018_';
// var wstart_date = '2017-01-01';
// var wend_date = '2019-01-01';



var outfile_name = outfile_name_prefix + wstart_date + "_" + wend_date;

print("Number of fiels in shapefile is ", SF.size());

//////////////////////////////////////////////////////
///////
///////     Functions
///////
//////////////////////////////////////////////////////
///
///  Function to clear clouds in Landsat-7
///
var cloudMaskL578_C2L2 = function(image) {
  var qa = image.select('QA_PIXEL');
  var cloud = qa.bitwiseAnd(1 << 3).and(qa.bitwiseAnd(1 << 9))
                .or(qa.bitwiseAnd(1 << 4)); // .and(qa.bitwiseAnd(1 << 11))
                //.or(qa.bitwiseAnd(1 << 5)).and(qa.bitwiseAnd(1 << 13)); // snow
  
  // Remove edge pixels that don't occur in all bands
  // var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()); //.updateMask(mask2);
};


////////////////////////////////////////////////
///
/// add Date of image to an imageCollection
///

function add_system_start_time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

function add_system_start_time_collection(colss){
 var c = colss.map(add_system_start_time_image);
 return c;
}

////////////////////////////////////////////////
///
///         NDVI - Landsat-8
///

function addNDVI_to_image(image) {
  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  return image.addBands(ndvi);
}

function add_NDVI_collection(image_IC){
  var NDVI_IC = image_IC.map(addNDVI_to_image);
  return NDVI_IC;
}


////////////////////////////////////////////////
///
///         EVI - Landsat-8
///

function add_EVI_collection(image_IC){
  var EVI_IC = image_IC.map(addEVI_to_image);
  return EVI_IC;
}

function addEVI_to_image(image) {
  var evi = image.expression(
                      '2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1.0))', {
                      'NIR': image.select('SR_B5'),
                      'RED': image.select('SR_B4'),
                      'BLUE':image.select('SR_B2')
                  }).rename('EVI');
  return image.addBands(evi);
}

////////////////////////////////////////////////
///
///         scale the bands
///
function scale_the_d_bands(image){
  var NIR = image.select('SR_B5').multiply(0.0000275).add(-0.2);
  var red = image.select('SR_B4').multiply(0.0000275).add(-0.2);
  var blue = image.select('SR_B2').multiply(0.0000275).add(-0.2);

  return image.addBands(NIR, null, true)
              .addBands(red, null, true)
              .addBands(blue, null, true);
}

////////////////////////////////////////////////
///
///         Do the Job function
///

function extract_satellite_IC(a_feature, start_date, end_date){
    var geom = a_feature.geometry(); // a_feature is a feature collection
    var newDict = {'original_polygon_1': geom};
    var imageC = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                .filterDate(start_date, end_date)
                .filterBounds(geom)
                .map(function(image){return image.clip(geom)})
                .sort('system:time_start', true);
    
    // scale the bands
    imageC = imageC.map(scale_the_d_bands);
    
    // toss out cloudy pixels
    imageC = imageC.map(cloudMaskL578_C2L2);
    
    // imageC = imageC.map(clean_clouds_from_one_image_landsat);
    imageC = add_NDVI_collection(imageC); // add NDVI as a band
    imageC = add_EVI_collection(imageC);  // add EVI as a band
    imageC = add_system_start_time_collection(imageC);
    
    // add original geometry to each image. We do not need to do this really:
    imageC = imageC.map(function(im){return(im.set(newDict))});
    
    // add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({ 'original_polygon': geom});
  return imageC;
}

function mosaic_and_reduce_IC_mean(an_IC,a_feature,start_date,end_date){
  an_IC = ee.ImageCollection(an_IC);
  var reduction_geometry = a_feature;
  var start_date_DateType = ee.Date(start_date);
  var end_date_DateType = ee.Date(end_date);
  //######**************************************
  // Difference in days between start and end_date

  var diff = end_date_DateType.difference(start_date_DateType, 'day');

  // Make a list of all dates
  var range = ee.List.sequence(0, diff.subtract(1)).map(function(day){
                                    return start_date_DateType.advance(day,'day')});

  // Funtion for iteraton over the range of dates
  function day_mosaics(date, newlist) {
    // Cast
    date = ee.Date(date);
    newlist = ee.List(newlist);

    // Filter an_IC between date and the next day
    var filtered = an_IC.filterDate(date, date.advance(1, 'day'));

    // Make the mosaic
    var image = ee.Image(filtered.mosaic());

    // Add the mosaic to a list only if the an_IC has images
    return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
  }

  // Iterate over the range to make a new list, and then cast the list to an imagecollection
  var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
  //print("newcol 1", newcol);
  //######**************************************

  var reduced = newcol.map(function(image){
                            return image.reduceRegions({
                                                        collection:reduction_geometry,
                                                        reducer:ee.Reducer.mean(), 
                                                        scale: 10
                                                      });
                                          }
                        ).flatten();
                          
  reduced = reduced.set({ 'original_polygon': reduction_geometry});
  
  return(reduced);
}

// remove geometry on each feature before printing or exporting

var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};

//////////////////////////////////////////////////////
///////
///////     Body
///////
//////////////////////////////////////////////////////

//////////////////////////////
///////
///////     Eastern WA
///////
//////////////////////////////
var xmin = -127.0;
var ymin = 45.35;
var xmax= -116.4;
var ymax = 49.1;
var xmed2 = (xmin + xmax) / 2.0;
var WA2 = ee.Geometry.Polygon([[xmed2, ymin], [xmed2, ymax], [xmax, ymax], [xmax, ymin], [xmed2, ymin]]);
var WA = [WA2];

var SF_regions = ee.FeatureCollection(WA);
var reduction_geometry = ee.FeatureCollection(SF);


Map.addLayer(SF_regions, {color: 'gray'}, 'WA');
Map.addLayer(SF, {color: 'blue'}, 'SF');
var ymed = (ymin + ymax)/2.0;
Map.setCenter(xmed2, ymed, 5);

// print ("Number of fields in the shapefile is", reduction_geometry.size());

var imageC = extract_satellite_IC(SF_regions, wstart_date, wend_date);
var reduced = mosaic_and_reduce_IC_mean(imageC, reduction_geometry, wstart_date, wend_date);  
var featureCollection = reduced;

Export.table.toDrive({
  collection: featureCollection,
  description:outfile_name,
  folder:"sixth_investig_intersected",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["ID", 'NDVI', 'EVI', "system_start_time"]
});



// Landsat 7

/*
 * https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2
 * 
 * Apparently collection 2 (USGS Landsat 7 Level 2, Collection 2, Tier 1 ) 
 * is supposed to be a better data compared to USGS Landsat 7 Surface Reflectance Tier 1  
 * (LANDSAT/LE07/C01/T1_SR)
 * 
 * That is how we found collection 2. But somehow my plots are not as good as T1_SR
 * 
 * There is no maskcloud() function on GEE developer page for collection 2.
 * So, I followed the same pattern in the mask functions in cloudMaskL457(.)
 * I replaced the right bands in that function (except I removed the edge-pixel-removal part). 
 * 
 * Moreover, there are more bitmasks for collection 2l snowy pixels, level of confidence of snow cover,
 * level of confidence of shadow cover. I do not know how we can take advantage of these if at all.
 * 
*/

// var SF = ee.FeatureCollection(intersect_Grant_2Cols);

// var SF = ee.FeatureCollection(intersect_Grant_Irr_2008_2018_2cols_10);
// var outfile_name_prefix = 'L7_T1C2L2_Scaled_int_Grant_Irr_2008_2018_2cols_10';
// var wstart_date = '2008-01-01';
// var wend_date = '2022-01-01';


// var SF = ee.FeatureCollection(Walla2015);
// var outfile_name_prefix = 'L7_T1C2L2_Scaled_Walla2015_';
// var wstart_date = '2014-01-01';
// var wend_date = '2016-01-01';

var SF = ee.FeatureCollection(AdamBenton2016);
var outfile_name_prefix = 'L7_T1C2L2_Scaled_AdamBenton2016_';
var wstart_date = '2015-01-01';
var wend_date = '2017-01-01';

// var SF = ee.FeatureCollection(Grant2017);
// var outfile_name_prefix = 'L7_T1C2L2_Scaled_Grant2017_';
// var wstart_date = '2016-01-01';
// var wend_date = '2018-01-01';

// var SF = ee.FeatureCollection(FranklinYakima2018);
// var outfile_name_prefix = 'L7_T1C2L2_Scaled_FranklinYakima2018_';
// var wstart_date = '2017-01-01';
// var wend_date = '2019-01-01';



var outfile_name = outfile_name_prefix + wstart_date + "_" + wend_date;
print (SF.first());

//////////////////////////////////////////////////////
///////
///////     Functions
///////
//////////////////////////////////////////////////////
///
///  Function to clear clouds in
///
var cloudMaskL578_C2L2 = function(image) {
  var qa = image.select('QA_PIXEL');
  var cloud = qa.bitwiseAnd(1 << 3).and(qa.bitwiseAnd(1 << 9))
                .or(qa.bitwiseAnd(1 << 4)); // .and(qa.bitwiseAnd(1 << 11))
                //.or(qa.bitwiseAnd(1 << 5)).and(qa.bitwiseAnd(1 << 13)); // snow
  // Remove edge pixels that don't occur in all bands
  // var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()); //.updateMask(mask2);
};

////////////////////////////////////////////////
///
/// add Date of image to an imageCollection
///

function add_system_start_time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

function add_system_start_time_collection(colss){
 var c = colss.map(add_system_start_time_image);
 return c;
}

////////////////////////////////////////////////
///
///         NDVI - Landsat-7
///

function addNDVI_to_image(image) {
  var ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI');
  return image.addBands(ndvi);
}

function add_NDVI_collection(image_IC){
  var NDVI_IC = image_IC.map(addNDVI_to_image);
  return NDVI_IC;
}


////////////////////////////////////////////////
///
///         EVI - Landsat-7
///
function addEVI_to_image(image) {
  var evi = image.expression(
                      '2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1.0))', {
                      'NIR': image.select('SR_B4'),
                      'RED': image.select('SR_B3'),
                      'BLUE': image.select('SR_B1')
                  }).rename('EVI');
  return image.addBands(evi);
}

function add_EVI_collection(image_IC){
  var EVI_IC = image_IC.map(addEVI_to_image);
  return EVI_IC;
}

////////////////////////////////////////////////
///
///         scale the bands
///
function scale_the_d_bands(image){
  var NIR = image.select('SR_B4').multiply(0.0000275).add(-0.2);
  var red = image.select('SR_B3').multiply(0.0000275).add(-0.2);
  var blue = image.select('SR_B1').multiply(0.0000275).add(-0.2);

  return image.addBands(NIR, null, true)
              .addBands(red, null, true)
              .addBands(blue, null, true);
}

////////////////////////////////////////////////
///
///         Do the Job function
///

function extract_satellite_IC(a_feature, start_date, end_date){
    var geom = a_feature.geometry(); // a_feature is a feature collection
    var newDict = {'original_polygon_1': geom};
    var imageC = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                .filterDate(start_date, end_date)
                .filterBounds(geom)
                .map(function(image){return image.clip(geom)})
                .sort('system:time_start', true);
    
    // scale the bands
    imageC = imageC.map(scale_the_d_bands);
    
    // toss out cloudy pixels
    imageC = imageC.map(cloudMaskL578_C2L2);
    
    // imageC = imageC.map(clean_clouds_from_one_image_landsat);
    imageC = add_NDVI_collection(imageC); // add NDVI as a band
    imageC = add_EVI_collection(imageC); // add EVI as a band
    imageC = add_system_start_time_collection(imageC);
    
    // add original geometry to each image. We do not need to do this really:
    imageC = imageC.map(function(im){return(im.set(newDict))});
    
    // add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({ 'original_polygon': geom});
  return imageC;
}

function mosaic_and_reduce_IC_mean(an_IC,a_feature,start_date,end_date){
  an_IC = ee.ImageCollection(an_IC);
  var reduction_geometry = a_feature;
  var start_date_DateType = ee.Date(start_date);
  var end_date_DateType = ee.Date(end_date);
  //######**************************************
  // Difference in days between start and end_date

  var diff = end_date_DateType.difference(start_date_DateType, 'day');

  // Make a list of all dates
  var range = ee.List.sequence(0, diff.subtract(1)).map(function(day){
                                    return start_date_DateType.advance(day,'day')});

  // Funtion for iteraton over the range of dates
  function day_mosaics(date, newlist) {
    // Cast
    date = ee.Date(date);
    newlist = ee.List(newlist);

    // Filter an_IC between date and the next day
    var filtered = an_IC.filterDate(date, date.advance(1, 'day'));

    // Make the mosaic
    var image = ee.Image(filtered.mosaic());

    // Add the mosaic to a list only if the an_IC has images
    return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
  }

  // Iterate over the range to make a new list, and then cast the list to an imagecollection
  var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
  //print("newcol 1", newcol);
  //######**************************************

  var reduced = newcol.map(function(image){
                            return image.reduceRegions({
                                                        collection:reduction_geometry,
                                                        reducer:ee.Reducer.mean(), 
                                                        scale: 10
                                                      });
                                          }
                        ).flatten();
                          
  reduced = reduced.set({ 'original_polygon': reduction_geometry});
  
  return(reduced);
}

// remove geometry on each feature before printing or exporting

var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};

//////////////////////////////////////////////////////
///////
///////     Body
///////
//////////////////////////////////////////////////////

//////////////////////////////
///////
///////     Eastern WA
///////
//////////////////////////////
var xmin = -127.0;
var ymin = 45.35;
var xmax= -116.4;
var ymax = 49.1;
var xmed2 = (xmin + xmax) / 2.0;
var WA2 = ee.Geometry.Polygon([[xmed2, ymin], [xmed2, ymax], [xmax, ymax], [xmax, ymin], [xmed2, ymin]]);
var WA = [WA2];

var SF_regions = ee.FeatureCollection(WA);
var reduction_geometry = ee.FeatureCollection(SF);

Map.addLayer(SF_regions, {color: 'gray'}, 'WA');
Map.addLayer(SF, {color: 'blue'}, 'SF');
var ymed = (ymin + ymax)/2.0;
Map.setCenter(xmed2, ymed, 5);

// print ("Number of fields in the shapefile is", reduction_geometry.size());

var imageC = extract_satellite_IC(SF_regions, wstart_date, wend_date);
var reduced = mosaic_and_reduce_IC_mean(imageC, reduction_geometry, wstart_date, wend_date);  
var featureCollection = reduced;

Export.table.toDrive({
  collection: featureCollection,
  description:outfile_name,
  folder:"sixth_investig_intersected",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["ID", 'NDVI', 'EVI', "system_start_time"]
});






