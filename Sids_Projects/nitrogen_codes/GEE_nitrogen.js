var SF = ee.FeatureCollection(corn_potato_2020_SF);
print("Number of fiels in shapefile is ", SF.size());

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
///
function addChl_to_image(image) {
  var cred = image.expression(
                      '(BB8 / BB6) - 1', {
                      'BB8': image.select('B8'),
                      'BB6': image.select('B6')
                  }).rename('CIRed');
  return image.addBands(cred);
}

function add_chl_collection(image_IC){
  var chl_IC = image_IC.map(addChl_to_image);
  return chl_IC;
}

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
///         Do the Job function
///

function extract_sentinel_IC(a_feature, start_date, end_date){
    var geom = a_feature.geometry(); // a_feature is a feature collection
    var newDict = {'original_polygon_1': geom};
    var imageC = ee.ImageCollection('COPERNICUS/S2') // Sentinel
                .filterDate(start_date, end_date)
                .filterBounds(geom)
                .map(function(image){return image.clip(geom)})
                .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))
                .sort('system:time_start', true);
    
    // toss out cloudy pixels
    imageC = imageC.map(maskS2clouds);
    
    // pick up some bands
    // imageC = imageC.select(['B8', 'B4', 'B3', 'B2']);
   
    imageC = add_system_start_time_collection(imageC);
    imageC = add_chl_collection(imageC);
    
    // add original geometry to each image
    // we do not need to do this really:
    imageC = imageC.map(function(im){return(im.set(newDict))});
    
    // add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({ 'original_polygon': geom,
                          'WSDA':a_feature
                        });
    // imageC = imageC.sort('system:time_start', true);
    //imageC = imageC.map(add_NDVI_collection)
  return imageC;
}

function mosaic_and_reduce_IC_mean(an_IC,a_feature,start_date,end_date){
  an_IC = ee.ImageCollection(an_IC);
  //print('mosaic_start_date:',start_date);
  //var reduction_geometry = ee.Feature(ee.Geometry(an_IC.get('original_polygon')));
  var reduction_geometry = a_feature;
  var WSDA = an_IC.get('WSDA');
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
                          
  reduced = reduced.set({ 'original_polygon': reduction_geometry,
                          'WSDA':WSDA
                      });
  WSDA = ee.Feature(WSDA);
  WSDA = WSDA.toDictionary();
  
  // var newDict = {'WSDA':WSDA};
  reduced = reduced.map(function(im){return(im.set(WSDA))}); 
  return(reduced);
}

// remove geometry on each feature before printing or exporting

var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};


var xmin=-125.0;
var ymin = 45.0;
var xmax=-116.0;
var ymax = 49.0;
var xmed1 = (xmin + xmax) / 2.0;
var xmed2 = (xmin + xmax) / 2.0;

var WA1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed1, ymax], [xmed1, ymin], [xmin, ymin]]);
var WA2 = ee.Geometry.Polygon([[xmed2, ymin], [xmed2, ymax], [xmax, ymax], [xmax, ymin], [xmed2, ymin]]);
var WA = [WA1,WA2];

var SF_regions = ee.FeatureCollection(WA);
var reduction_geometry = ee.FeatureCollection(SF);


Map.addLayer(SF_regions, {color: 'gray'}, 'WA');
Map.addLayer(reduction_geometry, {color: 'gray'}, 'reduction_geometry');

var wstart_date = '2020-01-01';
var wend_date = '2021-01-01';
var cloud_perc = 70;

var imageC = extract_sentinel_IC(SF_regions, wstart_date, wend_date);
var reduced = mosaic_and_reduce_IC_mean(imageC, reduction_geometry, wstart_date, wend_date);  
var featureCollection = reduced;

// var featureCollection = featureCollection.map(myproperties);

var outfile_name = 'Corn_Potato_Sent_' + wstart_date + "_" + wend_date;
Export.table.toDrive({
  collection: featureCollection,
  description:outfile_name,
  folder: "nitrogen",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["ID", "CropTyp", "CIRed", "system_start_time"]
});

