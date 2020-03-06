class World:

    def __init__(self, size_x, size_y, landmarks):
        """
        Initialize world with given dimensions.
        :param size_x: Length world in x-direction (m)
        :param size_y: Length world in y-direction (m)
        :param landmarks: List with 2D-positions of landmarks
        """

        self.x_max = size_x
        self.y_max = size_y

        print("Initialize world with landmarks {}".format(landmarks))

        # Check if each element is a list
        if any(not isinstance(lm, list) for lm in landmarks):

            # Not a list of lists, then exactly two elements allowed (x,y)-position single landmark
            if len(landmarks) != 2:
                print("Invalid landmarks provided to World: {}".format(landmarks))
            else:
                self.landmarks = [landmarks]
        else:
            # Check if each list within landmarks list contains exactly two elements (x- and y-position)
            if any(len(lm) != 2 for lm in landmarks):
                print("Invalid landmarks provided to World: {}".format(landmarks))
            else:
                self.landmarks = landmarks
