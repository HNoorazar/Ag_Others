

var SF = ee.FeatureCollection(SF_table);
print("Number of polygons in shapefile is", SF_table.size());

////////////////////////////////////////////////////////////////////////////////////////
///
///                           functions definitions start
///
////////////////////////////////////////////////////////////////////////////////////////
///
///  Function to mask clouds using the Sentinel-2 QA band.
///

function maskS2clouds(image) {
    var qa = image.select('QA60');

    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;

    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
                        qa.bitwiseAnd(cirrusBitMask).eq(0));

    // Return the masked and scaled data, without the QA bands.
    return image.updateMask(mask).divide(10000)
                          .select("B.*")
                          .copyProperties(image, ["system:time_start"]);
}

////////////////////////////////////////////////
///
/// add Year (Perhaps can be ommited and obtained in python from `system_start_time`)
///

function addYear_to_image(image){
  var year = image.date().get('year');
  var yearBand = ee.Image.constant(year).uint16().rename('image_year');
  return image.addBands(yearBand);
}

function addYear_to_collection(collec){
  var C = collec.map(addYear_to_image);
  return C;
}

////////////////////////////////////////////////
///
/// add Day of Year (Perhaps can be ommited and obtained in python from `system_start_time`)
///

function addDoY_to_image(image){
  // var doy = image.date().getRelative('day', 'year');
  var doy = ee.Date(image.get('system:time_start')).getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('doy');
  doyBand = doyBand.updateMask(image.select('B8').mask());
  return image.addBands(doyBand);
}

function addDoY_to_collection(collec){
  var C = collec.map(addDoY_to_image);
  return C;
}


////////////////////////////////////////////////
///
/// add Date of image to an imageCollection (Perhaps can be ommited and obtained in python from `system_start_time`)
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
///         EVI
///

function addEVI_to_image(image) {
  var evi = image.expression(
                      '2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1.0))', {
                      'NIR': image.select('B8'),
                      'RED': image.select('B4'),
                      'BLUE': image.select('B2')
                  }).rename('EVI');
  return image.addBands(evi);
}

function add_EVI_collection(image_IC){
  var EVI_IC = image_IC.map(addEVI_to_image);
  return EVI_IC;
}


////////////////////////////////////////////////
///
///         Do the Job function
///

function extract_sentinel_IC(a_feature,start_date, end_date){
    var geom = a_feature.geometry(); //a_feature is a feature collection
    var newDict = {'original_polygon_1': geom};
    var imageC = ee.ImageCollection('COPERNICUS/S2')
                .filterDate(start_date, end_date)
                .filterBounds(geom)
                // Pre-filter to get less cloudy granules.
                //.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", cloud_perc)
                .map(function(image){return image.clip(geom)})
                .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))
                //.filter('CLOUDY_PIXEL_PERCENTAGE < 70')
                .sort('system:time_start', true);
    
    
    imageC = imageC.map(maskS2clouds);      // toss out cloudy pixels
    imageC = addYear_to_collection(imageC); // add year as a band
    imageC = addDoY_to_collection(imageC);  // add DoY as a band
    imageC = add_EVI_collection(imageC);    // add EVI as a band
    
    imageC = add_system_start_time_collection(imageC);
    
    // add original geometry to each image
    // we do not need to do this really:
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
                                                        scale: 10//,
                                                        //tileScale: 16
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


// remove geometry on each feature before printing or exporting

var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};

////////////////////////////////////////////////////////////////////////////////////////
/////////
///////// Main Body
/////////

////////////////////////////////////////
/////
///// Idaho Polygon
/////
var xmin = -117.1;
var xmax = -111.3;

var ymin = 42.2;
var ymax = 43.93;

var idaho = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]);

////////////////////////////////////////
/////
///// WA and Oregon
/////

var xmin_WA_OR = -123.5;
var xmax_WA_OR = -117.5;
var xmed_WA_OR = (xmin_WA_OR + xmax_WA_OR) / 2.0;

var ymin_WA_OR = 44.4;
var ymax_WA_OR = 49.0;

var WA_OR_1 = ee.Geometry.Polygon([[xmin_WA_OR, ymin_WA_OR], 
                                   [xmin_WA_OR, ymax_WA_OR], 
                                   [xmed_WA_OR, ymax_WA_OR], 
                                   [xmed_WA_OR, ymin_WA_OR], 
                                   [xmin_WA_OR, ymin_WA_OR]]);
                                   
var WA_OR_2 = ee.Geometry.Polygon([[xmed_WA_OR, ymin_WA_OR], 
                                   [xmed_WA_OR, ymax_WA_OR], 
                                   [xmax_WA_OR, ymax_WA_OR], 
                                   [xmax_WA_OR, ymin_WA_OR], 
                                   [xmed_WA_OR, ymin_WA_OR]]);

////////////////////////////////////////
/////
/////   Northern California Polygon
/////
var xmin_NCali = -123.0;
var xmax_NCali = -118.0;

var ymin_NCali = 35.4;
var ymax_NCali = 39.0;

var North_Cali = ee.Geometry.Polygon([[xmin_NCali, ymin_NCali], 
                                      [xmin_NCali, ymax_NCali], 
                                      [xmax_NCali, ymax_NCali], 
                                      [xmax_NCali, ymin_NCali], 
                                      [xmin_NCali, ymin_NCali]]);


////////////////////////////////////////
/////
/////   Southern California Polygon
/////
var xmin_SCali = -116.5;
var xmax_SCali = -110.9;

var ymin_SCali = 32.0;
var ymax_SCali = 34.4;

var South_Cali = ee.Geometry.Polygon([[xmin_SCali, ymin_SCali], 
                                      [xmin_SCali, ymax_SCali], 
                                      [xmax_SCali, ymax_SCali], 
                                      [xmax_SCali, ymin_SCali], 
                                      [xmin_SCali, ymin_SCali]]);
////////////////////////////////////////
/////
/////  Colorado Polygon
/////
var xmin_Col = -107.0;
var xmax_Col = -105.5;

var ymin_Col = 37.0;
var ymax_Col = 38.0;

var Colorado = ee.Geometry.Polygon([[xmin_Col, ymin_Col], 
                                    [xmin_Col, ymax_Col], 
                                    [xmax_Col, ymax_Col], 
                                    [xmax_Col, ymin_Col], 
                                    [xmin_Col, ymin_Col]]);

////////////////////////////////////////
/////
/////  North Dakota Polygon
/////
var xmin_ND = -98.5;
var xmax_ND = -97.0;

var ymin_ND = 48.0;
var ymax_ND = 49.0;

var ND = ee.Geometry.Polygon([[xmin_ND, ymin_ND], 
                              [xmin_ND, ymax_ND], 
                              [xmax_ND, ymax_ND], 
                              [xmax_ND, ymin_ND], 
                              [xmin_ND, ymin_ND]]);


////////////////////////////////////////
/////
/////  Minnesota Polygon
/////
var xmin_MN = -96.5;
var xmax_MN = -92.3;

var ymax_MN = 47.0;
var ymin_MN = 43.3;

var MN = ee.Geometry.Polygon([[xmin_MN, ymin_MN], 
                              [xmin_MN, ymax_MN], 
                              [xmax_MN, ymax_MN], 
                              [xmax_MN, ymin_MN], 
                              [xmin_MN, ymin_MN]]);

////////////////////////////////////////
/////
/////  Wisconsin Polygon
/////
var xmin_Wisconsin = -90.0;
var xmax_Wisconsin = -88.0;

var ymax_Wisconsin = 45.6;
var ymin_Wisconsin = 43.3;

var Wisconsin = ee.Geometry.Polygon([[xmin_Wisconsin, ymin_Wisconsin], 
                                     [xmin_Wisconsin, ymax_Wisconsin], 
                                     [xmax_Wisconsin, ymax_Wisconsin], 
                                     [xmax_Wisconsin, ymin_Wisconsin], 
                                     [xmin_Wisconsin, ymin_Wisconsin]]);



////////////////////////////////////////
/////
/////  Michigan Polygon
/////
var xmin_MI = -86.0;
var xmax_MI = -84.4;

var ymax_MI = 43.6;
var ymin_MI = 41.5;

var Michigan = ee.Geometry.Polygon([[xmin_MI, ymin_MI], 
                                    [xmin_MI, ymax_MI], 
                                    [xmax_MI, ymax_MI], 
                                    [xmax_MI, ymin_MI], 
                                    [xmin_MI, ymin_MI]]);


////////////////////////////////////////
/////
/////  New York, New York, New Yooork Polygon
/////
var xmin_NY = -78.7;
var xmax_NY = -77.5;

var ymax_NY = 43.5;
var ymin_NY = 42.5;

var NYC = ee.Geometry.Polygon([[xmin_NY, ymin_NY], 
                               [xmin_NY, ymax_NY], 
                               [xmax_NY, ymax_NY], 
                               [xmax_NY, ymin_NY], 
                               [xmin_NY, ymin_NY]]);


////////////////////////////////////////
/////
/////  Maine Polygon
/////
var xmin_ME = -70.2;
var xmax_ME = -67.5;

var ymax_ME = 47.7;
var ymin_ME = 45.3;

var Maine = ee.Geometry.Polygon([[xmin_ME, ymin_ME], 
                                 [xmin_ME, ymax_ME], 
                                 [xmax_ME, ymax_ME], 
                                 [xmax_ME, ymin_ME], 
                                 [xmin_ME, ymin_ME]]);

////////////////////////////////////////
/////
/////  Florida Polygon
/////
var xmin_FL = -80.3;
var xmax_FL = -85.6;

var ymax_FL = 31.3;
var ymin_FL = 25.8;

var FL = ee.Geometry.Polygon([[xmin_FL, ymin_FL], 
                              [xmin_FL, ymax_FL], 
                              [xmax_FL, ymax_FL], 
                              [xmax_FL, ymin_FL], 
                              [xmin_FL, ymin_FL]]);



////////////////////////////////////////
/////
/////  Texas Polygon
/////
var xmin_TX = -98.7;
var xmax_TX = -97.6;

var ymax_TX = 27.3;
var ymin_TX = 25.8;

var TX = ee.Geometry.Polygon([[xmin_TX, ymin_TX], 
                              [xmin_TX, ymax_TX], 
                              [xmax_TX, ymax_TX], 
                              [xmax_TX, ymin_TX], 
                              [xmin_TX, ymin_TX]]);

//////////////////////////////////////////////////////////////////////////
var big_rectangle = [WA_OR_1, WA_OR_2, idaho, North_Cali, South_Cali, Colorado, ND, MN, 
                     Wisconsin, Michigan, NYC, Maine, FL, TX];


var SF_regions = ee.FeatureCollection(big_rectangle);
var reduction_geometry = ee.FeatureCollection(SF);

///////////////////
/////////////////// Visualize the rectangle
///////////////////
// Display the polygons by adding them to the map.
Map.centerObject(Colorado, 4);
Map.addLayer(SF, {color: 'blue'}, 'geodesic polygon');
Map.addLayer(idaho, {color: 'FF0000'}, 'geodesic polygon');
Map.addLayer(WA_OR_1, {color: 'green'}, 'geodesic polygon');
Map.addLayer(WA_OR_2, {color: 'green'}, 'geodesic polygon');

Map.addLayer(North_Cali, {color: 'yellow'}, 'geodesic polygon');
Map.addLayer(South_Cali, {color: 'red'}, 'geodesic polygon');

Map.addLayer(Colorado, {color: 'grey'}, 'geodesic polygon');

Map.addLayer(ND, {color: 'grey'}, 'geodesic polygon');
Map.addLayer(MN, {color: 'white'}, 'geodesic polygon');
Map.addLayer(Wisconsin, {color: 'green'}, 'geodesic polygon');
Map.addLayer(Michigan, {color: 'green'}, 'geodesic polygon');
Map.addLayer(NYC, {color: 'green'}, 'geodesic polygon');
Map.addLayer(Maine, {color: 'green'}, 'geodesic polygon');

Map.addLayer(FL, {color: 'green'}, 'geodesic polygon');
Map.addLayer(TX, {color: 'yellow'}, 'geodesic polygon');

var wstart_date = '2017-10-01';
var wend_date = '2017-12-31';
var cloud_perc = 70;

var imageC = extract_sentinel_IC(SF_regions, wstart_date, wend_date);
var reduced = mosaic_and_reduce_IC_mean(imageC, reduction_geometry, wstart_date, wend_date);  

var outfile_name = 'all_32_counties_1km_' + cloud_perc + 'cloud';
Export.table.toDrive({
  collection: reduced,
  description:outfile_name,
  folder:"Supriya",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["grid_id", "doy", "EVI",
             "system_start_time", "image_year", "B8"]
});








