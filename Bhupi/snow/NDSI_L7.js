// %%
/*

// %%
Here I want to look at NDSI: Normalized Difference Snow Index
This index is sensitive, i.e. may not work well on mountains
since some sort of cloud can be mistake for snow;
"Remote sensing, hydrological modeling and in situ observations in snow cover research: A review"

// %%
Perhaps that sort of cloud will be present when there is snow, i.e.
during particular time of the year. So, whenever there is no snow, then
perhaps we can use NDSI. or perhaps bare soil index, or soil-moisture indices?

// %%
Given that Bhupi gave me an example from 2013 I assume that year or perhaps
prior years are of interest. So, here I will do LANDSAT; i.e. not sentinel.

// %%
https://www.usgs.gov/landsat-missions/normalized-difference-snow-index

// %%
*/

// %% [markdown]
// print (coordinates_tb.first().get("Station_Name"));
// print (coordinates_tb.first().geometry().coordinates());
// print (ee.Number(coordinates_tb.first().geometry().coordinates().get(0)));
// print (coordinates_tb.first().geometry().coordinates().get(1));

// %% [markdown]
// Map.addLayer(coordinates_tb, {color: 'red'}, 'coordinates');
// Map.setCenter(-120, 46.0, 6);

// %%
var data_src = 'LANDSAT/LE07/C02/T1_L2';

// %% [markdown]
// ///////////////////////////////////////////////////////////
// //////
// //////   Functions
// //////

// %%
function create_rectangle(a_feat_) {
  // Create a rectangle around each coordinate in the 
  // "coordinates_tb" table which is the CSV file from Bhupi.
  var a_point_ = ee.Geometry.Point(a_feat_.geometry().coordinates());
  var round_buffer = a_point_.buffer({'distance': 3000});
  var a_rectan_ = ee.Feature(round_buffer.bounds());
  
  var Station_Name = a_feat_.get("Station_Name");
  a_rectan_ = a_rectan_.set({'Station_Name': Station_Name});
  return a_rectan_;
}

// %%
function create_rectangle_4_allPoints(a_feat_coll_){
  var all_rect_ = ee.FeatureCollection(a_feat_coll_.map(create_rectangle));
  return (all_rect_);
}

// %%
var cloudMaskL578_C2L2 = function(image) {
  var qa = image.select('QA_PIXEL');
  var cloud = qa.bitwiseAnd(1 << 3).and(qa.bitwiseAnd(1 << 9))
                .or(qa.bitwiseAnd(1 << 4)); // .and(qa.bitwiseAnd(1 << 11))
                //.or(qa.bitwiseAnd(1 << 5)).and(qa.bitwiseAnd(1 << 13)); // snow
  
  // Remove edge pixels that don't occur in all bands
  // var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()); //.updateMask(mask2);
};


// %% [markdown]
// //////////////////////////////////////////////
// /
// / add Date of image to an imageCollection
// /

// %%
function add_system_start_time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

// %%
function add_system_start_time_collection(colss){
 var c = colss.map(add_system_start_time_image);
 return c;
}
////////////////////////////////////////////////
///
///         NDSI - Landsat-7
///
function add_NDSI_collection(image_IC){
  var NDVI_IC = image_IC.map(addNDSI_to_image);
  return NDVI_IC;
}

// %%
function addNDSI_to_image(image) {
  var ndsi = image.normalizedDifference(['SR_B2', 'SR_B5']).rename('NDSI');
  return image.addBands(ndsi);
}
////////////////////////////////////////////////
///
///       LSWI
///

// %%
function add_LSWI_to_image(image) {
  var LSWI = image.normalizedDifference(['SR_B4', 'SR_B5']).rename('LSWI');
  return image.addBands(LSWI);
}

// %%
function add_LSWI_collection(image_IC){
  var LSWI_IC = image_IC.map(add_LSWI_to_image);
  return LSWI_IC;
}
////////////////////////////////////////////////
///
///  https://www.sciencedirect.com/science/article/pii/S034181622200251X
///  Normalized Difference bare soil index

// %%

// %%
////////////////////////////////////////////////
///
/// BSI = ((Red+SWIR) - (NIR+Blue)) / ((Red+SWIR) + (NIR+Blue)) 
///
function add_BSI_collection(image_IC){
  var BSI_IC = image_IC.map(addBSI_to_image);
  return BSI_IC;
}

// %%
function addBSI_to_image(image) {
  var bsi = image.expression(
                      '((RED+SWIR) - (NIR+BLUE)) / ((RED+SWIR) + (BLUE))', {
                      'NIR': image.select('SR_B4'),
                      'RED': image.select('SR_B3'),
                      'BLUE':image.select('SR_B1'),
                      'SWIR':image.select('SR_B5')
                  }).rename('BSI');
  return image.addBands(bsi);
}

// %%
////////////////////////////////////////////////
///
///         scale the damn bands
///
function scale_the_damn_bands(image){
  var NIR = image.select('SR_B4').multiply(0.0000275).add(-0.2);
  var red = image.select('SR_B3').multiply(0.0000275).add(-0.2);
  var blue = image.select('SR_B1').multiply(0.0000275).add(-0.2);
  var green = image.select('SR_B2').multiply(0.0000275).add(-0.2);
  var SWI1 = image.select('SR_B5').multiply(0.0000275).add(-0.2);
  
  return image.addBands(NIR, null, true)
              .addBands(red, null, true)
              .addBands(blue, null, true)
              .addBands(green, null, true)
              .addBands(SWI1, null, true);
}

// %% [markdown]
// //////////////////////////////////////////////
// /
// /         Do the Job function
// /

// %%
function extract_satellite_IC(a_feature, start_date, end_date){
    var geom = a_feature.geometry(); // a_feature is a feature collection
    var newDict = {'original_polygon_1': geom};
    var imageC = ee.ImageCollection(data_src)
                   .filterDate(start_date, end_date)
                   .filterBounds(geom)
                   .map(function(image){return image.clip(geom)})
                   .sort('system:time_start', true);
    
    imageC = imageC.map(scale_the_damn_bands); // scale the damn bands
    imageC = imageC.map(cloudMaskL578_C2L2);   // toss out cloudy pixels
    
    // imageC = imageC.map(clean_clouds_from_one_image_landsat);
    imageC = add_NDSI_collection(imageC);
    imageC = add_BSI_collection(imageC);
    imageC = add_LSWI_collection(imageC);
    // imageC = addNDBSI_to_IC(imageC);
    
    imageC = add_system_start_time_collection(imageC);

    // add original geometry to each image. We do not need to do this really:
    imageC = imageC.map(function(im){return(im.set(newDict))});
    
    // add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({ 'original_polygon': geom});
  return imageC;
}

// %%
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

// %% [markdown]
// remove geometry on each feature before printing or exporting

// %%
var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};

// %% [markdown]
// ///////////////////////////////////////////////////////////
// //////
// //////   Driver Body.
// //////

// %%
var recs_outFunc = create_rectangle_4_allPoints(coordinates_tb);
// print (recs_outFunc);
Map.addLayer(recs_outFunc, {'color': 'red'}, 'red: all rectangles');
Map.setCenter(-120, 46.0, 6);

// %%
//////////////////////////////
///////
///////     Eastern WA
///////
//////////////////////////////
var xmin =-124.0;
var ymin = 45.5;
var xmax = -113.0;
var ymax = 49.0;
var xmed1 = (xmin + xmax) / 2.0;
var WA1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed1, ymax], [xmed1, ymin], [xmin, ymin]]);
var WA2 = ee.Geometry.Polygon([[xmed1, ymin], [xmed1, ymax], [xmax, ymax], [xmax, ymin], [xmed1, ymin]]);

// %%
var xmin2 =-124.0;
var ymin2 = 41.8;
var xmax2 = -113.0;
var ymax2 = 45.5;
var xmed2 = (xmin2 + xmax2) / 2.0;
var R3 = ee.Geometry.Polygon([[xmin2, ymin2], [xmin2, ymax2], [xmed2, ymax2], [xmed2, ymin2], [xmin2, ymin2]]);
var R4 = ee.Geometry.Polygon([[xmed2, ymin2], [xmed2, ymax2], [xmax2, ymax2], [xmax2, ymin2], [xmed2, ymin2]]);

// %%
var WA = [WA1, WA2, R3, R4];

// %%
var SF_regions = ee.FeatureCollection(WA);
var reduction_geometry = ee.FeatureCollection(recs_outFunc);


// %%
Map.addLayer(SF_regions, {color: 'gray'}, 'WA');
var ymed = (ymin + ymax)/2.0;
Map.setCenter(xmed1, ymed, 5);

// %%
// print ("Number of fields in the shapefile is", reduction_geometry.size());
var wstart_date = '2013-01-01';
var wend_date   = '2013-12-30';

// %%
var imageC = extract_satellite_IC(SF_regions, wstart_date, wend_date);
var reduced = mosaic_and_reduce_IC_mean(imageC, reduction_geometry, wstart_date, wend_date);  
var featureCollection = reduced;
var outfile_name = "NDSI_A2_L7";
Export.table.toDrive({
  collection: featureCollection,
  description:outfile_name,
  folder:"snow",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["Station_Name", "NDSI", "BSI", "LSWI", "system_start_time"]
});
