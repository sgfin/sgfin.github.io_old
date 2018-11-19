//
// ** jQuery Layout Functions
//
// ** Provided by www.pumkin.co.uk
//
// ** BETA 2nd March 2011
//
// ** Re-use as desired but please retain this credit
//
// ** Requires jQuery 1.4+
//
//
//
//
// 
// Resizes an element to fit or fill its container
// option params - sendWid, sendHei - override the auto wid/hei detection 
// (useful for images which haven't loaded yet, if you already know the size)
//
function fitArea(target, fillorfit, centreItem, sendWid, sendHei, margins) {
  //
  var originalWid = sendWid > 0 ? sendWid : target.width();
  var originalHei = sendHei > 0 ? sendHei : target.height();
  var wid = target.parents('div').first().width();
  var hei = target.parents('div').first().height();
  target.width(wid);
  var scaleRatio = wid/originalWid;
  target.height(Math.round(originalHei*scaleRatio));
  if (fillorfit == 'fit') {
    if (target.height() > hei) {
      scaleRatio = hei/originalHei;
      target.height(hei);
      target.width(Math.round(originalWid*scaleRatio));
    }
  } else {
    if (target.height() < hei) {
      scaleRatio = hei/originalHei;
      target.height(hei);
      target.width(Math.round(originalWid*scaleRatio));
    }
  }
  if (centreItem == true) {
    position(target); 
  }
}

//
// Resizes the image by setting the dimensions on the tag as paramters
//
function setDimensionsTofitArea(target, maxWid, maxHei, getWid, getHei, margins) {
  //
  if (!margins || margins.length<4) {
    margins = new Array(0,0,0,0);
  } 
  var xOffset = margins[3]+margins[1];
  var yOffset = margins[0]+margins[2];
  maxWid -= xOffset;
  maxHei -= yOffset;
  //
  var scaleRatio = maxWid/getWid;
  var newHei = Math.round(getHei*scaleRatio);
  if (newHei > maxHei) {
    scaleRatio = maxHei/getHei;
    newWid = Math.round(getWid*scaleRatio);
    target.attr("width", newWid);
    target.attr("height", maxHei);
  } else {
    target.attr("width", maxWid);
    target.attr("height", newHei);  
  }
  //
  if (margins[0] > 0) {
    target.css("padding-top", margins[0]);
    target.css("padding-right", margins[1]);
    target.css("padding-bottom", margins[2]);
    target.css("padding-left", margins[3]);
  }
}

//
// Positions content within its container
// can send margins as an array of 4 numbers (top, right, bottom, left)
// and align as an array of 2 items, eg ["top","left"]
//
function position(target, margins, align, animate) {
  //
  if (!margins || margins.length<4) {
    margins = new Array(0,0,0,0);
  }
  var container = target.parents('div').first();
  var areaWidth = container.width();
  var areaHeight = container.height();
  var wid = target.width();
  var hei = target.height();
  //
  var xOffset = margins[3]-margins[1];
  var yOffset = margins[0]-margins[2];
  //  
  var targetX = null;
  var targetY = null;
  //
  // if margins and align parameters sent, align to one side, centre on the other axis
  if (align && align.length<2) {
    align.push(""); 
  }
  if (align) {
    switch (align[0]) {
      case "right" :
        var targetX = areaWidth - (margins[1] + wid);
        break;
      case "bottom" :
        var targetY = areaHeight - (margins[2] + hei);
        break;
      case "left" :
        var targetX = margins[3];
        break;
      case "top" :
        var targetY = margins[0];
        break;
      // otherwise align centre 
      default : 
        var targetX = ((areaWidth + xOffset) / 2) - (wid / 2);
        var targetY = ((areaHeight + yOffset) / 2) - (hei / 2);
        break;
    }
    switch (align[1]) {
      case "right" :
        var targetX = areaWidth - (margins[1] + wid);
        break;
      case "bottom" :
        var targetY = areaHeight - (margins[2] + hei);
        break;
      case "left" :
        var targetX = margins[3];
        break;
      case "top" :
        var targetY = margins[0];
        break;  
      // otherwise align centre 
      default : 
        if (!targetX && targetX !== 0) { var targetX = ((areaWidth + xOffset) / 2) - (wid / 2); }
        if (!targetY && targetY !== 0) { var targetY = ((areaHeight + yOffset) / 2) - (hei / 2); }
        break;
    }
  } 
  //otherwise centre within margins
  else {
    var targetX = ((areaWidth + xOffset) / 2) - (wid / 2);
    var targetY = ((areaHeight + yOffset) / 2) - (hei / 2);   
  }
  //
  if (animate==true) {
    //  
    $(target).animate(
      {
      "margin-left": Math.round(targetX),
      "margin-top": Math.round(targetY)
      },
      "medium"
    );
  } else {
  
    target.css("margin-left", Math.round(targetX));
    target.css("margin-top", Math.round(targetY));
  }
}

