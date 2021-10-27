class Student():
    weekly_schedule = {0:[1,2,3,4,5,6,7]}

    def get_todays_schedule(self, day):
        print(Student.weekly_schedule[day])

me = Student()
me.get_todays_schedule(0)
