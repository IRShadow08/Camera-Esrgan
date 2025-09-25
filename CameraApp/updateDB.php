<?php
require __DIR__ . '/database.php';

header('Content-Type: application/json');
$data = json_decode(file_get_contents("php://input"), true);

// Validate input
if (!isset($data['course_id']) || !isset($data['records'])) {
    echo json_encode(['success' => false, 'error' => 'Missing course_id or records']);
    exit;
}

$course_id = $data['course_id'];
$records = $data['records'];
$inserted = 0;

foreach ($records as $record) {
    $name = $record['name'];
    $timestamp = $record['timestamp'];

    $parts = explode(' ', $name, 2);
    $firstname = $parts[0] ?? '';
    $lastname = $parts[1] ?? '';

    $res = pg_query_params($con,
        "SELECT id FROM student WHERE firstname ILIKE $1 AND lastname ILIKE $2 LIMIT 1",
        [$firstname, $lastname]
    );

    $student = pg_fetch_assoc($res);
    if (!$student) continue;

    $student_id = $student['id'];

    $res = pg_query_params($con,
        "INSERT INTO record (student, date) VALUES ($1, $2) RETURNING id",
        [$student_id, $timestamp]
    );
    $record_row = pg_fetch_assoc($res);
    $record_id = $record_row['id'] ?? null;

    if ($record_id) {
        pg_query_params($con,
            "INSERT INTO class (record, course, attendance, sent) VALUES ($1, $2, $3, $4)",
            [$record_id, $course_id, 1, date('Y-m-d H:i:s')]
        );
        $inserted++;
    }
}

echo json_encode(['success' => true, 'inserted' => $inserted]);
