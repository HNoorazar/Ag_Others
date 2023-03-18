

var xmin = -122.0 ;
var xmax= -120.2;

var ymin = 35.35;
var ymax = 37.2;

var xmed2 = (xmin + xmax) / 2.0;
var ROI = ee.Geometry.Polygon([[xmed2, ymin], [xmed2, ymax], [xmax, ymax], [xmax, ymin], [xmed2, ymin]]);
ROI = [ROI];
ROI = ee.FeatureCollection(ROI);


var data_source = 'LANDSAT/LE07/C02/T1_L2';
var start_date = '2014-01-01';
var end_date = '2014-01-10';

var imageC = ee.ImageCollection(data_source)
               .filterDate(start_date, end_date)
               .filterBounds(ROI)
               .map(function(image){return image.clip(ROI)})
               .sort('system:time_start', true);
               
print (imageC.size());


/*
 The following for-loops will not work.
*/
var outfile_name_prefix = "LANDSAT7_C02_T1_L2_";
for (var n=0; n < imageC.size().getInfo(); n++) {
  var image = ee.Image(imageC.get(n));
  
  Export.image.toDrive({
    image: image,
    description: outfile_name_prefix + n,
    folder: 'snapshot_Monterey',
    region: ROI,
    scale: 10,
    maxPixels: 1e10,
  });
}



/*
   Documentation for the following package is here:
   https://gis.stackexchange.com/questions/248216/exporting-each-image-from-collection-in-google-earth-engine
   https://github.com/fitoprincipe/geetools-code-editor/wiki/Batch
*/
var batch = require('users/fitoprincipe/geetools:batch');
batch.Download.ImageCollection.toDrive(imageC, "folder_name_here", 
                                      {name: '{system_date}',
                                      scale: 30,
                                      region: ROI.getInfo(), 
                                      type: 'float'
                                      });
