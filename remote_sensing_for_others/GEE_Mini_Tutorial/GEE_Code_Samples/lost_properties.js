/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var ROI = 
    /* color: #98ff00 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[100.14445909748613, 14.689307337967634],
          [100.14445909748613, 14.04009719681829],
          [100.98491319904863, 14.04009719681829],
          [100.98491319904863, 14.689307337967634]]], null, false);
/***** End of imports. If edited, may not auto-convert in the playground. *****/
/*
This is the Answer to the following question
https://gis.stackexchange.com/questions/360313/lost-properties-in-imagecollection-after-mosaicking-image-collection-in-google-e
*/
var ROI = 
    ee.Geometry.Polygon(
        [[[100.14445909748613, 14.689307337967634],
          [100.14445909748613, 14.04009719681829],
          [100.98491319904863, 14.04009719681829],
          [100.98491319904863, 14.689307337967634]]], null, false);

Map.centerObject(ROI, 10);

var start = ee.Date('2017-09-03');
var finish = ee.Date('2017-10-30');

var S1 = ee.ImageCollection('COPERNICUS/S1_GRD')
           .filterDate(start, finish)
           .filterBounds(ROI)
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
           .filter(ee.Filter.eq('instrumentMode', 'IW'));

print('S1',S1)

/////////////////////Mosaicking a Image Collection by Date (day) ////////////////

// Difference in days between start and finish
var diff = finish.difference(start, 'day');

// Make a list of all dates
var range = ee.List.sequence(0, diff.subtract(1)).map(function(day){return start.advance(day,'day')});

// Funtion for iteraton over the range of dates
var day_mosaics = function(date, newlist) {
  // Cast
  date = ee.Date(date);
  newlist = ee.List(newlist);
  
  // Filter collection between date and the next day
  var filtered = S1.filterDate(date, date.advance(1,'day'));
  
  var timeBand = ee.Image(date.millis())
    .divide(1000 * 3600 * 24) // Unix epoch days
    .int()
    .rename('t')
    
  // Make the mosaic
  var image_m = ee.Image(filtered.mosaic())
    .addBands(timeBand)
  
  // Add the mosaic to a list only if the collection has images
  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image_m), newlist))
}

// Iterate over the range to make a new list, and then cast the list to an imagecollection
var S1_mosaic = ee.ImageCollection(ee.List(range.iterate(day_mosaics,ee.List([]))));

print('S1_mosaic',S1_mosaic)

var linearFit = S1_mosaic.select(['t', 'VH'])
                         .reduce(ee.Reducer.linearFit());
  
print("linearFit", linearFit)
Map.addLayer(linearFit, {bands: 'scale', min: -0.5, max: 0.5}, 'scale')
Map.addLayer(linearFit, {bands: 'offset', min: -3000, max: 9000}, 'offset')
