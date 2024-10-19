CREATE TABLE IF NOT EXISTS t_schedule (
    id INT AUTO_INCREMENT PRIMARY KEY,
#     user_id INT NOT NULL,
    date DATE NOT NULL,
    description VARCHAR(255),
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

# test data
INSERT INTO t_schedule (date, description) VALUES
('2024-10-01', 'Prepare work report'),
('2024-10-02', 'Attend team meeting'),
('2024-10-03', 'Complete project documentation'),
('2024-10-04', 'Conduct code review'),
('2024-10-05', 'Client feedback meeting'),
('2024-10-06', 'Team building activity')