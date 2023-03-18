/**
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
 * Source: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
 */
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}


/**
 * Function to mask clouds based on the pixel_qa band of Landsat 8 SR data.
 * @param {ee.Image} image Input Landsat 8 SR image
 * @return {ee.Image} Cloudmasked Landsat 8 image
 * 
 * Source: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T2_SR#description
 */
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask);
}


/**
 * Function to mask clouds based on the pixel_qa band of Landsat SR data.
 * @param {ee.Image} image Input Landsat SR image
 * @return {ee.Image} Cloudmasked Landsat image
 * 
 * Source: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C01_T1_SR
 */
var cloudMaskL457 = function(image) {
  var qa = image.select('pixel_qa');
  // If the cloud bit (5) is set and the cloud confidence (7) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 5)
                  .and(qa.bitwiseAnd(1 << 7))
                  .or(qa.bitwiseAnd(1 << 3));
  // Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};





/**
 * 
 * This is for masking clouds in Landsat-7 Level-2 data product: LANDSAT/LE07/C02/T1_L2
 * I wrote this that masks clouds, cloud shadows, and snows.
 * In 4-fields that I tried the snow filter (the line .or(qa.bitwiseAnd(1 << 5)).and(qa.bitwiseAnd(1 << 13));) 
 * did not make much of a difference.
 * 
 * I wrote this by looking into the function cloudMaskL457(.) which
 * I am not sure what is the logic of the script or how the logical operators (AND and OR)
 * (e.g. the order they are applied) works. 
 * 
 * Therefore, I do not guarantee this function! However, in my 
 * experiment (4 fields over the years 2008-2021; whatever range in there that data was available)
 * the NDVI computations, using cloudMaskL7_level2(.) for masking pixels, performed poorly
 * I am not sure wether the poor performace was effect of my function or Level-2 data.
 * 
 */

var cloudMaskL7_level2 = function(image) {
  var qa = image.select('QA_PIXEL');
  // If the cloud bit (5) is set and the cloud confidence (7) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 3).and(qa.bitwiseAnd(1 << 9))
                .or(qa.bitwiseAnd(1 << 4)).and(qa.bitwiseAnd(1 << 11));
                //.or(qa.bitwiseAnd(1 << 5)).and(qa.bitwiseAnd(1 << 13));
  // Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};


/**
 * I am trying the following now (Aug. 27. 4:17 PM)
 * The output name starts with Grant_4Fields_Landsat7_T1_L2_onlyFirstThreeMaskElemeents_
 * 
 */

var cloudMaskL7_level2 = function(image) {
  var qa = image.select('QA_PIXEL');
  // If the cloud bit (5) is set and the cloud confidence (7) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 3).and(qa.bitwiseAnd(1 << 9))
                .or(qa.bitwiseAnd(1 << 4)); // .and(qa.bitwiseAnd(1 << 11))
                //.or(qa.bitwiseAnd(1 << 5)).and(qa.bitwiseAnd(1 << 13));
  // Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};
