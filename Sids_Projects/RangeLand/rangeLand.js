
var SF = ee.FeatureCollection(rangeLandSF);
var numeric_start_year = 2000;
var numeric_end_year = 2005;

var wstart_date = numeric_start_year + '-01-01';
var wend_date = numeric_end_year + '-01-01';


// var outfile_name_prefix = 'L7_T1C2L2_Scaled_rangeLand_';
var outfile_name = wstart_date + "_" + wend_date;
print (SF.first());
print ("Number of polygons in the shapefile is ", SF.size());

//////////////////////////////////////////////////////
///////
///////     Functions
///////
//////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////
//////////
//////////     Landsat 8
//////////
// From https://developers.google.com/earth-engine/guides/ic_composite_mosaic
// Define a function that scales and masks Landsat 8 surface reflectance images.
function prepSrL8(image) {
  // Develop masks for unwanted pixels (fill, cloud, cloud shadow).
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);

  // Apply the scaling factors to the appropriate bands.
  var getFactorImg = function(factorNames) {
    var factorList = image.toDictionary().select(factorNames).values();
    return ee.Image.constant(factorList);
  };
  var scaleImg = getFactorImg([
    'REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B10']);
  var offsetImg = getFactorImg([
    'REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B10']);
  var scaled = image.select('SR_B.|ST_B10').multiply(scaleImg).add(offsetImg);

  // Replace original bands with scaled bands and apply masks.
  return image.addBands(scaled, null, true)
    .updateMask(qaMask).updateMask(saturationMask);
}


// This function masks clouds and adds quality bands to Landsat 8 images.
var addQualityBands_L8 = function(image) {
  // Normalized difference vegetation index.
  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']);
  // Image timestamp as milliseconds since Unix epoch.
  // var millis = ee.Image(image.getNumber('system:time_start')).rename('system_time_start').toFloat();
  return prepSrL8(image).addBands([ndvi]); //millis
};


//////////////////////////////////////////
///////
///////   Landsat 7
///////

function prepSrL7(image) {
  // Develop masks for unwanted pixels (fill, cloud, cloud shadow).
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);

  // Apply the scaling factors to the appropriate bands.
  var getFactorImg = function(factorNames) {
    var factorList = image.toDictionary().select(factorNames).values();
    return ee.Image.constant(factorList);
  };
  var scaleImg = getFactorImg(['REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B6']);
  var offsetImg = getFactorImg(['REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B6']);
  var scaled = image.select('SR_B.|ST_B6').multiply(scaleImg).add(offsetImg);

  // Replace original bands with scaled bands and apply masks.
  return image.addBands(scaled, null, true)
              .updateMask(qaMask).updateMask(saturationMask);
}

// This function masks clouds and adds quality bands to Landsat 7 images.
var addQualityBands_L7 = function(image) {
  var ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']);
  return prepSrL8(image).addBands([ndvi]); 
};


//////////////////////////////////////////
///////
///////   Landsat 5
///////

function prepSrL5(image) {
  // Develop masks for unwanted pixels (fill, cloud, cloud shadow).
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);

  // Apply the scaling factors to the appropriate bands.
  var getFactorImg = function(factorNames) {
    var factorList = image.toDictionary().select(factorNames).values();
    return ee.Image.constant(factorList);
  };
  var scaleImg = getFactorImg(['REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B6']);
  var offsetImg = getFactorImg(['REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B6']);
  var scaled = image.select('SR_B.|ST_B6').multiply(scaleImg).add(offsetImg);

  // Replace original bands with scaled bands and apply masks.
  return image.addBands(scaled, null, true)
              .updateMask(qaMask).updateMask(saturationMask);
}

// This function masks clouds and adds quality bands to Landsat 7 images.
var addQualityBands_L5 = function(image) {
  var ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']);
  return prepSrL8(image).addBands([ndvi]); 
};


function maxQualityMosaic_by_month(ICC){
  var months = ee.List.sequence(1, 12);
  var years = ee.List.sequence(numeric_start_year, numeric_end_year);
  var byMonthYear = ee.ImageCollection.fromImages(
  years.map(function(y) {
    return months.map(function (m) {
      return ICC.filter(ee.Filter.calendarRange(y, y, "year"))
                .filter(ee.Filter.calendarRange(m, m, "month"))
                .qualityMosaic("nd")
                .set("month", m).set("year", y);
                     });
                           }).flatten());
  return byMonthYear;
}


function addMonthYear_toBands(image) {
  image = image.addBands(image.metadata('month'));
  image = image.addBands(image.metadata('year'));
  return image;
}

function addMonthYear_toBands_Coll(colss){
 var c = colss.map(addMonthYear_toBands);
 return c;
}


function mosaic_and_reduce_IC_mean(an_IC, a_feature, start_date, end_date){
  an_IC = ee.ImageCollection(an_IC);
  var reduction_geometry = a_feature;
  
  var start_date_DateType = ee.Date(start_date);
  var end_date_DateType = ee.Date(end_date);
  var diff = end_date_DateType.difference(start_date_DateType, 'day');
  var range = ee.List.sequence(0, diff.subtract(1)).map(function(day){
                                  return start_date_DateType.advance(day,'day')});
  // print ("an_IC", an_IC);
  // print ("range", range);
  function day_mosaics(date, newlist) {
    date = ee.Date(date);
    newlist = ee.List(newlist);
    var filtered = an_IC.filterDate(date, date.advance(1, 'day'));
    var image = ee.Image(filtered.mosaic());
    return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
  }
  var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
  // print ("newcol", newcol);
  var reduced = newcol.map(function(image){
                            return image.reduceRegions({collection:reduction_geometry,
                                                        reducer:ee.Reducer.mean(), 
                                                        scale: 10});
                                          }
                        ).flatten();
  
  // var reduced = an_IC.map(function(image){
  //                           return image.reduceRegions({collection:reduction_geometry,
  //                                                       reducer:ee.Reducer.mean(), 
  //                                                       scale: 10});
  //                                         }
  //                       ).flatten();
  
  reduced = reduced.set({ 'original_polygon': reduction_geometry});
  return(reduced);
}


////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
var xmin = -125.5;
var xmax = -65.3;
var xmed1 = (xmin + xmax) / 2.0;

var ymax = 49.5;
var ymin = 23.3;
var ymed1 = (ymin + ymax) / 2.0;

var box1 = ee.Geometry.Polygon([[xmin, ymed1], [xmin, ymax], [xmed1, ymax], [xmed1, ymed1], [xmin, ymed1]]);
var box2 = ee.Geometry.Polygon([[xmed1, ymed1], [xmed1, ymax], [xmax, ymax], [xmax, ymed1], [xmed1, ymed1]]);
var box3 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymed1], [xmed1, ymed1], [xmed1, ymin], [xmin, ymin]]);
var box4 = ee.Geometry.Polygon([[xmed1, ymin], [xmed1, ymed1], [xmax, ymed1], [xmax, ymin], [xmed1, ymin]]);
var big_rectangle = [box1, box2, box3, box4];

var SF_regions = ee.FeatureCollection(big_rectangle);


// var greenestPixelComposite_L8 = collection_L8.qualityMosaic('nd');

// Load a 2014 Landsat 8 ImageCollection.
// Map the cloud masking and quality band function over the collection.

var geom = SF_regions.geometry(); // a_feature is a feature collection
var collection_L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                      .filterDate(wstart_date, wend_date)
                      .map(addQualityBands_L8)
                      .filterBounds(geom)
                      .map(function(image){return image.clip(geom)})
                      .sort('system:time_start', true);


var collection_L7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                      .filterDate(wstart_date, wend_date)
                      .map(addQualityBands_L7)
                      .filterBounds(geom)
                      .map(function(image){return image.clip(geom)})
                      .sort('system:time_start', true);
                   

var collection_L5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
                      .filterDate(wstart_date, wend_date)
                      .map(addQualityBands_L5)
                      .filterBounds(geom)
                      .map(function(image){return image.clip(geom)})
                      .sort('system:time_start', true);


collection_L8 = collection_L8.select(['nd']);
collection_L7 = collection_L7.select(['nd']);
collection_L5 = collection_L5.select(['nd']);

var L78 = ee.ImageCollection(collection_L8.merge(collection_L7));
var L578 = ee.ImageCollection(L78.merge(collection_L5));


var monthly_greenestPixelComposite_L578 = maxQualityMosaic_by_month(L578);
// print (L578.size());


var vvv = addMonthYear_toBands_Coll(monthly_greenestPixelComposite_L578);
// print ("vvv", vvv);

var reduced = vvv.map(function(image){
                      return image.reduceRegions({collection:SF,
                                                 reducer:ee.Reducer.mean(), 
                                                 scale: 30});
                                          }).flatten();
reduced = reduced.set({ 'original_polygon': SF});
// print (reduced.first());

var outfile_name_prefix="qualityMosaicAttempt1_scale_30_L578_";
var outfile_name = outfile_name_prefix + wstart_date + "_" + wend_date;

Export.table.toDrive({
  collection: reduced,
  folder:"rangeLand",
  description: outfile_name,
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["lat", "long", "nd", "month", "year"]
});

