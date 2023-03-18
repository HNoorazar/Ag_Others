// Differece between this ans A1_S1_monthly_practice is 
// that here we do not use mosaic and in A1_S1 we do. See what is the diff.
// It seems there are less clouds here! July is clear in both tho.
// Let us toss images with more than 20% cloud in them. (80% cleam)
// and then try to toss cloudy pixels.
// var image = IC_fetched_scaled.filterDate(start, end).median();
// var image = IC_mosaiced.filterDate(start, end).median();
// 
//
var wYear=2018;
var data_source = 'COPERNICUS/S2';
var wstart_date = wYear + '-05-01'; // YYYY-MM-DD
var wend_date   = wYear + '-09-01'; // YYYY-MM-DD

var xmin = -120.5;
var ymin = 46.0;
var xmax = -116.9;
var ymax = 48.0;
var xmed = (xmin + xmax) / 2.0;

var WA_rect = ee.Geometry.Rectangle([xmin, ymin, xmed, ymax]);
var WA_rect_name = "WA1_";

// var WA_rect = ee.Geometry.Rectangle([xmed, ymin, xmax, ymax]);
// var WA_rect_name = "WA2_";

// var big_rect = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax]);
var big_rect = WA_rect;
Map.addLayer(big_rect, {color: 'gray'}, 'WA');


var ymed = (ymin + ymax) / 2.0;
Map.setCenter(xmed, ymed, 7);

////////////////////////////////////////////////////////
//////
//////    Functions
//////

function maskS2clouds(image) {
    var qa = image.select('QA60');

    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;

    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
               qa.bitwiseAnd(cirrusBitMask).eq(0));

    // Return the masked and scaled data, without the QA bands.
    return image.updateMask(mask)
                .select("B.*")
                .copyProperties(image, ["system:time_start"]);
}


function add_system_start_time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

function add_system_start_time_collection(colss){
 var c = colss.map(add_system_start_time_image);
 return c;
}

function choose_4bands(image){
  //
  // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
  // All are in 10 meter resolution
  // B2 is blue
  // B3 is green
  // B4 is red
  // B8 is NIR
  //
  return image.select(['B2', 'B3', 'B4', 'B8'], ['blue', 'green', 'red', 'NIR']);
}

// var img = ee.Image('COPERNICUS/S2_SR/20210109T185751_20210109T185931_T10SEG');
// print (img);
// img = scale_bands_and_choose_4bands(img);


function fetch(data_source, start_date, end_date, AOI){
  var an_IC = ee.ImageCollection(data_source)
                .filterDate(wstart_date, wend_date)
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", 20)
                .filterBounds(AOI)
                .map(function(image){return image.clip(AOI)})
                .sort('system:time_start', true);
                
  an_IC = an_IC.map(maskS2clouds);
                
  // scale the damn bands
  // an_IC = an_IC.map(choose_4bands);
  // print ("an_IC", an_IC.first());

  an_IC = add_system_start_time_collection(an_IC);
  return an_IC;
}

function mosaic_IC(an_image_coll, start_date, end_date, AOI){
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

    // Filter an_image_coll between date and the next day
    var filtered = an_image_coll.filterDate(date, date.advance(1, 'day'));

    var image = ee.Image(filtered.mosaic()); // Make the mosaic
    
    image = image.copyProperties({source: filtered.first(),
                                  properties: ['system:time_start']
                                  });
    image = image.set({system_time_start: filtered.first().get('system:time_start')});

    // Add the mosaic to a list only if the an_image_coll has images
    return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
  }

  // Iterate over the range to make a new list, and then cast the list to an imagecollection
  var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
  newcol = newcol.sort('system_time_start', true);
  return(newcol);
}

function visualize_and_export_image(image2vis){
  var imageRGB_mosaiced1 = image2vis.visualize(vizParams);
  
  imageRGB_mosaiced1 = imageRGB_mosaiced1.copyProperties({source: image2vis,
                                                          properties: ['system:time_start']
                                                          });
  imageRGB_mosaiced1 = imageRGB_mosaiced1.set({system_time_start: image2vis.get('system_time_start')});
  // imageRGB_mosaiced1 = imageRGB_mosaiced1.addBands(image2vis.get('system_time_start'));
  return imageRGB_mosaiced1;
}


/////////////////////////////////////////////
/////
/////      Body
/////
/////////////////////////////////////////////
// var neededBands = ['blue', 'green', 'red', 'NIR'];
var neededBands = ['B2', 'B3', 'B4', 'B8'];
  
var IC_fetched = fetch(data_source, wstart_date, wend_date, big_rect);
// print ("IC_fetched.size(): ", IC_fetched.size());
// print ("IC_fetched.first() is ", IC_fetched.first());


function scale_bands_and_choose_4bands(image){
  //
  // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
  // All are in 10 meter resolution
  // B2 is blue
  // B3 is green
  // B4 is red
  // B8 is NIR
  //
  var bands = ['B2', 'B3', 'B4', 'B8'];
  var new_bands = ['blue', 'green', 'red', 'NIR'];
  // return image.select(bands).rename(new_bands);
    
  return image.divide(10000)
              // .select(['B2', 'B3', 'B4', 'B8'], ['blue', 'green', 'red', 'NIR'])
              .select(bands).rename(new_bands)
              .copyProperties(image, ["system:time_start"]);
}

// subset and scale bands
var IC_fetched_scaled = IC_fetched.map(scale_bands_and_choose_4bands);
// print ("IC_fetched_scaled: ", IC_fetched_scaled.first());
// print ("IC_fetched_scaled size is: ", IC_fetched_scaled.size());

var IC_mosaiced = mosaic_IC(IC_fetched_scaled, wstart_date, wend_date, big_rect);
// print ("line 190, IC_mosaiced.size is ", IC_mosaiced.size());
var months = ee.List.sequence(5, 8);
var monthly_composite_median = ee.ImageCollection(months.map(function(m) {
    var start = ee.Date.fromYMD(wYear, m, 1);
    var end = start.advance(1, 'month');
    var image = IC_fetched_scaled.filterDate(start, end).median();
    // var image = IC_mosaiced.filterDate(start, end).median();
    return image.set('month', m);
}));

// print ("line 211. monthly_composite_median.size() is ", monthly_composite_median.size());
// print ("line 212. monthly_composite_median is ", monthly_composite_median); 

var vizParams = {
  bands: ["red", "green", "blue"],
  min: 0,
  max: 0.2, // https://developers.google.com/earth-engine/tutorials/tutorial_api_04 
  // gamma: [0.1, 1.1, 1]
};

//Add the Image
Map.addLayer(monthly_composite_median.first(), vizParams);
// Map.addLayer(WA_rect, {color: 'red'}, 'WA1_rect');

//
// Export the damn thing.
//

var size_=monthly_composite_median.size();
var monthly_composite_median_list = monthly_composite_median.toList(size_);
// print ("Line 231. monthly_composite_median_list is ", monthly_composite_median_list);
// print ("Line 232. monthly_composite_median_list.get(1) is ", monthly_composite_median_list.get(1));
// print (ee.Image(monthly_composite_median_list.get(1)).get("system_time_start"));
// print ((ee.Image(monthly_composite_median_list.get(0)).get("month")).getInfo());

var outfile_name_prefix = wYear;
var output_folder = "A1_S2_monthly_practice";

var client_size = size_.getInfo();
print ("client_size is ", client_size);
for (var n=0; n < client_size; n++) {
  var image = ee.Image(monthly_composite_median_list.get(n));
  var damn_time = image.get('month');
  Export.image.toDrive({
    image: image,
    description: WA_rect_name + "month_" + damn_time.getInfo() +  "_year_" + outfile_name_prefix,
    folder: output_folder,
    scale: 10,
    region: big_rect,
    maxPixels: 1e13,
  });
}

