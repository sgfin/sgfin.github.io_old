// JavaScript Document

function jumpTo(id) {
  //
  $.scrollTo('#'+id, 1200, {offset:-30});   
}

//$(document).ready(function(){
//             $(".tweet").tweet({
//                    //join_text: "auto",
//          modpath: 'http://nicolapotts.com/twitter/index.php',
//                    username: "nicolapottscom",
//          page:1,
//                    count: 3,
//                    loading_text: "loading tweets...",
//          template: "{text}"
//          //template: "<p>{text}</p><p>{time} &bull; {reply_action} &bull; {retweet_action} &bull; {favorite_action}</p>"
//          //auto_join_text_default: "I said:",        // [string]   auto text for non verb: "i said" something
//          //auto_join_text_ed: "I",                   // [string]   auto text for past tense: "i" surfed
//          //auto_join_text_ing: "I am",               // [string]   auto tense for present tense: "i was" surfing
//          //auto_join_text_reply: "I replied to",     // [string]   auto tense for replies: "i replied to" @someone "with"
//          //auto_join_text_url: "I was looking at",   // [string]   auto tense for urls: "i was looking at" http:...
//        });/*.bind("loaded", function(){
//          $(this).find("a.tweet_action").click(function(ev) {
//          window.open(this.href, "Retweet",
//                'menubar=0,resizable=0,width=550,height=420,top=200,left=400');
//          ev.preventDefault();
//          });
//        });*/
//            });

/*
jQuery(function($){
      $("#custom").tweet({
        avatar_size: 32,
        count: 4,
        username: "seaofclouds",
        template: "{text} � {retweet_action}"
      });
    }).bind("loaded", function(){
      $(this).find("a.tweet_action").click(function(ev) {
        window.open(this.href, "Retweet",
                    'menubar=0,resizable=0,width=550,height=420,top=200,left=400');
        ev.preventDefault();
      });
    });
*/


function getWinSize() {
  
  var w = 0;
  var h = 0;

  w = $(window).width();
  h = $(window).height();
  
  return [w,h];
}


function findPos(obj) {
  var curleft = curtop = 0;
  if (obj.offsetParent) {
    do {
      curleft += obj.offsetLeft;
      curtop += obj.offsetTop;
    } while (obj = obj.offsetParent);
    return [curleft,curtop];
  }
}



function setPosition(element1, element2) {
  
  // where el1 is to be positioned relative to el2
  el1 = document.getElementById(element1);
  el2 = document.getElementById(element2);
  
  // get window size
  var winDimensions = getWinSize();
  var winWidth = winDimensions[0];
  var winHeight = winDimensions[1];
  
  el1.style.display = 'block';
  
  // get element size (info, in this case)
  var el1Width = el1.clientWidth; //getElementWidth(el1);
  var el1Height = el1.clientHeight; //getElementHeight(el1);
  
  // get element position (thumb in this case)
  var el2Pos = findPos(el2);
  var el2Left = el2Pos[0];
  var el2Top = el2Pos[1];
    
  el1.style.position = 'absolute';  
    
  if (winWidth < (el2Left + 140 + 365)) {
    el1.style.left = '-366px';
  }
  //alert('winheight = '+winHeight+' : space required = '+(el1Height));
  if (winHeight < (el2Top + el1Height)) {
    
    el1.style.top = '-'+(el1Height - 118)+'px';
  }
  
  //el1.style.visibility = 'visible';
  el2.style.zIndex = 999;
}


function restorePosition(element1, element2) {
  
  el1 = document.getElementById(element1);
  el2 = document.getElementById(element2);
  el1.style.display = 'none';
  //el1.style.visibility = 'hidden';
  el1.style.position = 'fixed';
  el1.style.left = '140px';
  el1.style.top = '0px';
  el2.style.zIndex = 10;
}
