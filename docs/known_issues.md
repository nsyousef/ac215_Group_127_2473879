# Known Issues/Changes

## Frontend - T

* when you try to scroll up when it is generating text (after adding disease), it auto scrolls you to the bottom
* put time tracking on right and chat in middle
* put generic chat in Results panel and put LLM output in chat only
* make images show up in time tracking after disease initially added (most recent should be at top, as it is currently)
* add dates to cases
* add ability to delete cases
* Fix disease list:
  * show disease, date, image
    * date should reflect date of latest uploaded image
* Chat doesn't load when you go to an existing disease from the home screen
* When navigating to the results/track/chat page, make sure all cards are scrolled to the top (or bottom for the chat)
* For body map selection:
  * add more options to list (e.g. ear, armpit, etc.)
  * add ability to turn body around to show back
  * remove distinction between left and right

## Frontend performance - N

* download trained model file from GCP bucket on first startup
* lazy import PyTorch when running an ML model
* try to optimize Python bundle

## Training

* get a better model (the current one sucks!) - JR

## Inference

* if the model is not confident about anything, return "unknown" - R

## Deployment

* CI/CD
* Pulumi/Kubernetes deployment (for cloud stuff) (oh boy!) - JR
* set up CI pipeline to auto build DMG images on push to main - N
* demonstrate scaling (e.g. simulate many users calling our cloud API) - JR
* Deploy updates to the Kubernetes cluster upon merging changes into the main branch. - JR
* Validation checks to ensure only models meeting performance thresholds are deployed. - JR

## Other Milestone 5

* blog - N
* video
* Demo Examples/Walkthrough
* clean up code and make sure it is documented
