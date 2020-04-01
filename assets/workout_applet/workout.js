// Note: The codepen version of this timer does not have sound, as I cannot host assets. 
// Sorry.
// to-do: Mute option buttons if interval running
// combine interval functions
// Add "only works with JavaScript"
// Make options text editable


// Helpers by Sam
function getAllCheckboxes(groupsList){
    return groupsList.map(function(group_name)
        {return Array.from(document.getElementsByName(group_name + "_Exercises"), item => item)}).flat()
    }

// Helpers by Sam
function getCheckedExerciseNames(group_name){
    return Array.from(document.getElementsByName(group_name + "_Exercises"), item => item).
        filter(function(el) {return el.checked}).
        map(function(x) {return x.getAttribute("exercise_name")})
    }

function getExerciseCounts(group_name){
    return Array.from(document.getElementsByName(group_name + "_n"), item => item).
        map(function(x) {return parseInt(x.value) }).
        map(function(x) {return Math.min(x, getCheckedExerciseNames(group_name).length)})
    }

// Helpers 
function getRandom(arr, n) {
    var result = new Array(n),
        len = arr.length,
        taken = new Array(len);
    if (n > len)
        throw new RangeError("getRandom: more elements taken than available");
    while (n--) {
        var x = Math.floor(Math.random() * len);
        result[n] = arr[x in taken ? taken[x] : x];
        taken[x] = --len in taken ? taken[len] : len;
    }
    return result;
}

function randSort(a, b) {  
  return 0.5 - Math.random();
}

function sampleExercises(groupsList){
    items = groupsList.filter(function(x) {return getExerciseCounts(x) > 0}).
        map( x => getRandom(getCheckedExerciseNames(x), getExerciseCounts(x))).
    flat()
    return Array.from(new Set(items)).sort(randSort).concat("Rest");
}



var app = angular.module("tabataApp", []);
app.controller("tabataAppCtrl", ["$scope", function($scope) {

  
/* ===========================*/


// groupsList = [
// sampleExercises(groupsList)


  /* ===============================*/

  //    Set default volume as on
  // $("#volume-switch").prop("checked", true);

  $scope.workouts = {
    "Pull": ["Pull-Ups", "Inverted Rows", "1 Arm Negs", "Archer Pull-Ups", "Frenchies", "Slow Pull-Ups"],
    "Push": ["Ring Push-Ups", "Clap Push-Ups", "Archer Push-Ups", "Spiderman Push-Ups", "Kinesin Push-Ups", "Pike Push-Ups", "Dips"],
    "Legs": ["Jump Squats", "Single Leg Squats", "1 Arm DB Swing", "DB Lunge", "Wood Chop", "Single-Leg RDL", "Front Squats", "DB Hang Clean"],
    "Abs": ["Ab Mat", "Plank", "Around the Worlds", "Hanging Leg Lift"],
    "Arms": ["Burpies", "Cross-Body Hammer", "Upright Row", "Arnolds", "Overhead Tricep"]
  }

  $scope.groupsList = ["Pull", "Push", "Legs", "Abs", "Arms"]
  $scope.workout = sampleExercises($scope.groupsList)
  $scope.current_round = $scope.workout.slice();

  $scope.rounds = {
    roundsLeft: 1,
    totalRounds: 3
  };

  $scope.optionTimes = {
    timeOff: "00:15",
    timeOn: "00:45"
  };

  $scope.timerTimes = {
    breakTime: "00:15",
    workTime: "00:45"
  };

  $scope.timerStates = {
    breakRunning: false,
    workRunning: false
  };


  /* ===============================*/

  // Parse time string to integers
  function parseTime(time) {
    var time = time.split(":");
    var SECONDS = parseInt(time[1]);
    var MINUTES = parseInt(time[0]);
    return [MINUTES, SECONDS];
  }

  // Put 0 in front if digit less than 10
  function minTwoDigits(n) {
    return (n < 10 ? "0" : "") + n;
  }

  function makeSound(currentSeconds) {
      if (currentSeconds >= 2) {
        if (currentSeconds == 46){
            var endBeep = new buzz.sound("./workout_applet/sound_effects/45.wav").play();
        }
        else if (currentSeconds == 31){
            var endBeep = new buzz.sound("./workout_applet/sound_effects/30.wav").play();
        }
        else if (currentSeconds == 21){
            var endBeep = new buzz.sound("./workout_applet/sound_effects/20.wav").play();
        }
        else if (currentSeconds == 11){
            var endBeep = new buzz.sound("./workout_applet/sound_effects/10.wav").play();
        }
        else if (currentSeconds == 6){
            var endBeep = new buzz.sound("./workout_applet/sound_effects/5.wav").play();
        }
      } else {
        var endBeep = new buzz.sound("./workout_applet/sound_effects/tone.mp3").play();
      }
   }

  // Adds or removes 1 second
  $scope.changeTime = function(currentTime, deltaTime) {
    var time = parseTime(currentTime);
    var minutes = time[0];
    var seconds = time[1];
    var newTime = "";
    // If interval is running, make sound
     if (seconds <= 50 && minutes === 0 && $("#volume-switch").prop("checked") && ($scope.timerStates.breakRunning || $scope.timerStates.workRunning)) {
      makeSound(seconds);
     }
    if (seconds === 59 && deltaTime === "+1") {
      newTime = (minTwoDigits(minutes + 1) + ":" + "00").toString();
    } else if (minutes >= 1 && seconds === 0 && deltaTime === "-1") {
      newTime = (minTwoDigits(minutes - 1) + ":" + "59").toString();
    } else if (minutes === 0 && seconds === 0 && deltaTime === "-1") {
      newTime = (minTwoDigits(minutes) + ":" + minTwoDigits(seconds)).toString();
    } else {
      var tempTime = minTwoDigits(eval(seconds + deltaTime));
      newTime = (minTwoDigits(minutes) + ":" + tempTime).toString();
    }
    return newTime;
  };

  /* Functions to change the option numbers
  ================================================*/

  // Change the number of rounds
  $scope.changeRounds = function(currentRounds, deltaRounds) {
    if ($scope.rounds.totalRounds === 0 && deltaRounds === "-1") {
      return;
    }
    $scope.rounds.totalRounds = eval(currentRounds + deltaRounds);
  };

  // Could not find a way to pass property, so this is the hacky temporary solution
  $scope.changeTimeOff = function(currentTime, deltaTime) {
    var newTime = $scope.changeTime(currentTime, deltaTime);
    $scope.optionTimes.timeOff = newTime;
    $scope.timerTimes.breakTime = newTime;
  };

  $scope.changeTimeOn = function(currentTime, deltaTime) {
    var newTime = $scope.changeTime(currentTime, deltaTime);
    $scope.optionTimes.timeOn = newTime;
    $scope.timerTimes.workTime = newTime;
  };

  // Switch between break screen and work screen
  function switchScreens(value) {
    if (value === "toWork") {
      $("#time-left").removeClass("hidden");
      $("#break-left").addClass("hidden");
      $("#current-timer").css("background-color", "#a5d6a7");
      $scope.timerStates.workRunning = true;
    } else {
      $("#break-left").removeClass("hidden");
      $("#time-left").addClass("hidden");
      $("#current-timer").css("background-color", "#ef9a9a");
      $scope.timerStates.breakRunning = true;
    }
    $scope.startClock();
  } // End switchScreens

  // Start the timer
  $scope.startClock = function() {
    $("#pause-button").removeClass("hidden");
    $("#start-button").addClass("hidden");
    // If work is showing on click, change state
    if ($("#break-left").hasClass("hidden")) {
      $scope.timerStates.workRunning = true;
    }

    // If there is a round left
    if ($scope.rounds.roundsLeft <= $scope.rounds.totalRounds) {
      if (!$scope.timerStates.workRunning) {
        breakInterval = setInterval(function() {
          $scope.timerStates.breakRunning = true;
          var newTime = $scope.changeTime($scope.timerTimes.breakTime, "-1");
          $scope.timerTimes.breakTime = newTime;
          if (newTime === "00:00") {
            stopCurrentInterval();
            $scope.timerStates.breakRunning = false;
            var temp = $scope.optionTimes.timeOff;
            $scope.timerTimes.breakTime = temp;
            $scope.$apply();
            switchScreens("toWork");
          }
          $scope.$apply();
        }, 1000);
      } else if (!$scope.timerStates.breakRunning) {
        workInterval = setInterval(function() {
          $scope.timerStates.workRunning = true;
          var newTime = $scope.changeTime($scope.timerTimes.workTime, "-1");
          $scope.timerTimes.workTime = newTime;
          if (newTime === "00:00") {
            stopCurrentInterval();
            $scope.timerStates.workRunning = false;
            var temp = $scope.optionTimes.timeOn;
            $scope.timerTimes.workTime = temp;
            $scope.$apply();
            if($scope.current_round.length > 1){
            $scope.current_round.shift();
                }
            else{
            $scope.rounds.roundsLeft++;
            $scope.current_round = $scope.workout.slice();
            }
            switchScreens("toBreak");
          }
          $scope.$apply();
        }, 1000);
      }
    } else {
      $("#pause-button").addClass("hidden");
      $("#start-button").removeClass("hidden");
      $scope.clear();
    }
  }; // End startClock()

  // Clear whichever interval is running
  function stopCurrentInterval() {
    if ($scope.timerStates.workRunning) {
      $scope.timerStates.workRunning = false;
      clearInterval(workInterval);
    } else {
      $scope.timerStates.breakRunning = false;
      clearInterval(breakInterval);
    }
  }

  $scope.pauseClock = function() {
    $("#pause-button").addClass("hidden");
    $("#start-button").removeClass("hidden");
    stopCurrentInterval();
  };

  $scope.clear = function() {
    $scope.timerTimes.breakTime = $scope.optionTimes.timeOff;
    $scope.timerTimes.workTime = $scope.optionTimes.timeOn;
    $scope.rounds.roundsLeft = 1;
    $("#break-left").removeClass("hidden");
    $("#time-left").addClass("hidden");
    $("#current-timer").css("background-color", "#ef9a9a");
    $scope.pauseClock();
  };

  $scope.generateWorkout = function(){
    $scope.workout = sampleExercises($scope.groupsList);
    $scope.current_round = $scope.workout.slice();
  };

  $scope.uncheckAllBoxes = function(){
    var allboxes = getAllCheckboxes($scope.groupsList);
    for (var i = 0; i < allboxes.length; i++) { 
            allboxes[i].checked = false; 
        } 
  };

  $scope.checkAllBoxes = function(){
    var allboxes = getAllCheckboxes($scope.groupsList);
    for (var i = 0; i < allboxes.length; i++) { 
            allboxes[i].checked = true; 
        } 
  };


}]); // End controller